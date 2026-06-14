# Constitution of India — RAG

Local Retrieval-Augmented Generation over the **official** Constitution of India
(Ministry of Law and Justice, Legislative Department — "As on 1st May, 2026").

Pipeline:

```
query
  └─ embed (bge-base-en-v1.5)
       └─ retrieve top-K from BOTH collections (naive + structured)
            └─ merge + dedupe
                 └─ rerank (bge-reranker-v2-m3 cross-encoder)
                      └─ top-N passages
                           └─ legal LLM (SaulLM-7B) with graded, cited answers
```

## Two chunking strategies (both indexed)

- **naive** — fixed char windows over the whole document (100% coverage by design).
- **structured** — Article/Part/Schedule-aware chunks with citation metadata. It
  *partitions* the text (contiguous, gap-free), so even a missed header only
  merges text into a neighbour — nothing is skipped.

Hybrid retrieval queries **both**, so the naive index is a safety net for
anything the structured chunker mis-segments; the reranker then picks the best.

## Models

| role | model | runs via |
|------|-------|----------|
| embedding | `BAAI/bge-base-en-v1.5` | sentence-transformers (GPU/CPU) |
| reranker | `BAAI/bge-reranker-v2-m3` | sentence-transformers CrossEncoder |
| inference | `SaulLM-7B-Instruct` (legal) | Ollama |

The inference prompt is **graded**: it answers from the retrieved context and
cites Articles precisely, and only falls back to the model's own legal knowledge
when context is insufficient — clearly labeled `General knowledge (not in
retrieved text - verify):` and never with fabricated citations.

## Layout

```
constitution-rag/
├── data/                 # Constitution of India.pdf (+ cached constitution.txt)
├── backend/
│   ├── config.py         # all settings (env-overridable)
│   ├── extract.py        # PDF -> text (pdftotext / pypdf; no OCR)
│   ├── chunkers.py       # naive + structured chunkers (offset-based)
│   ├── ingest.py         # build the two ChromaDB collections
│   ├── compare.py        # coverage proof + chunk stats
│   ├── dump.py           # write chunks to preview/ for inspection
│   ├── embeddings.py     # bge-base embedding backend
│   ├── rerank.py         # bge-reranker cross-encoder
│   ├── retrieve.py       # hybrid retrieve + rerank
│   └── query.py          # answer() with graded prompt (+ CLI)
├── ui/app.py             # streamlit chat (hybrid/rerank toggles)
├── scripts/
│   ├── Modelfile.saul    # Ollama Modelfile for SaulLM-7B
│   └── setup_saul.sh     # download GGUF + `ollama create`
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd 05-labs/constitution-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # needs poppler-utils for pdftotext
cp .env.example .env

# Legal LLM (downloads a GGUF, registers it with Ollama as saul-7b-instruct):
bash scripts/setup_saul.sh
```

> First use also downloads the bge embedding + reranker weights from Hugging Face
> (~440 MB + ~600 MB). Install a CUDA `torch` build to use your GPU.

## Workflow

```bash
export PYTHONPATH=.

# 1. Inspect chunking before embedding (no models needed).
python -m backend.compare --sample
python -m backend.dump --article 21

# 2. Build both vector collections (uses bge-base embeddings).
python -m backend.ingest                 # --mode naive|structured|both, --force

# 3. Ask questions (hybrid + rerank by default).
python -m backend.query                  # add --show to see retrieved passages
python -m backend.query --no-hybrid --strategy structured
python -m backend.query --no-rerank

# 4. Web UI.
streamlit run ui/app.py

# 5. Measure retrieval quality across configs (structured/naive/hybrid/+rerank).
python -m backend.evaluate                # Hit@1/3/5, MRR, Recall@K
python -m backend.evaluate --k 10 --verbose

# 6. Answer-level eval (runs the LLM): citation/faithfulness/keyword/abstention.
python -m backend.evaluate --with-llm
python -m backend.evaluate --with-llm --judge qwen2.5:7b-instruct
```

## Evaluation

`eval/questions.json` holds labeled questions (factual / scenario / negative)
with expected Article number(s) and key phrases.

**Retrieval eval** (`backend.evaluate`) is retrieval-only (fast, deterministic,
no LLM): Hit@1/3/5, MRR, Recall@K per config. Chunks are credited by *character
position* (via the structured article spans), so naive and structured are judged
fairly. Use it to tune `TOP_N`, `RERANK_*`, hybrid on/off with evidence.

**Answer eval** (`--with-llm`) runs the production pipeline and scores the
generated answers:

- **citation_hit** — does the answer cite the expected Article?
- **faithfulness** — every Article it cites must be in the retrieved passages
  (flags hallucinated citations).
- **keyword_recall** — fraction of expected key phrases present.
- **abstention** — for negative questions, did it correctly hedge / label
  general knowledge instead of inventing an Article?
- **--judge** (optional) — an LLM-as-judge scores correctness + groundedness 1-5.

Every `--with-llm` run writes a full transcript (questions, retrieved passages,
answers, metrics, judge notes) to `eval/runs/run_<timestamp>.json` for review.

## Tuning (see `.env.example`)

- Retrieval: `RAG_HYBRID`, `RETRIEVE_K`, `TOP_N`, `RERANK_ENABLED`.
- Embeddings: `EMBEDDING_MODEL`, `EMBED_DEVICE`, `EMBED_BATCH`
  (changing the model requires re-running `backend.ingest`).
- Chunking: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_ARTICLE_CHARS`.
- LLM: `OLLAMA_MODEL`, `OLLAMA_HOST`.

## Notes

- Both collections share one `chroma_db/` and use cosine space.
- Text extraction is cached to `data/constitution.txt`; use `--force` to rebuild.
- No OCR: the PDF has a real text layer (English). The Hindi pages are images.
