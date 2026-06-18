# Constitution of India — RAG

Local Retrieval-Augmented Generation over the **official** Constitution of India
(Ministry of Law and Justice, Legislative Department — "As on 1st May, 2026").

Pipeline:

```
user message (+ optional chat history)
  └─ query rewrite (follow-ups → standalone retrieval query)
       └─ embed (bge-base-en-v1.5)
            └─ hybrid retrieve (naive + structured, top-K each)
                 └─ merge + dedupe
                      └─ rerank (bge-reranker-v2-m3 cross-encoder)
                           └─ top-N passages → graded LLM draft (Ollama)
                                └─ citation critic (rewrite or safe fallback)
                                     └─ streamed / displayed answer
```

**Highlights**

- **Zero-skip ingestion** — two independent chunkers, both proven 100% coverage.
- **Hybrid retrieval + cross-encoder rerank** over a naive *and* a structured index.
- **Graded, cited answers** that quote Articles and label any general-knowledge fallback.
- **Conversational chat** — follow-ups like "what are its exceptions?" are rewritten
  using recent history before retrieval (first message unchanged).
- **Citation critic** — post-generation faithfulness check; rewrites answers that
  cite Articles absent from retrieved text, or falls back to a safe excerpt.
- **Streaming UI** — Streamlit chat with example prompts, clear history, persistent
  sources and retrieved passages.
- **Measured, not vibes** — a 34-question eval harness scoring retrieval *and* answer
  quality (citation, faithfulness, abstention, LLM-as-judge). See [Results](#results).
- **Runs out of the box** — the official Constitution PDF is included in `data/`.

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
| inference (default) | `qwen2.5:7b-instruct` | Ollama |
| inference (optional) | `SaulLM-7B-Instruct` (legal-tuned) | Ollama |

The default inference model is `qwen2.5:7b-instruct` (the one the results below
were measured with). A legal-tuned alternative, **SaulLM-7B**, can be registered
via `scripts/setup_saul.sh` and selected with `OLLAMA_MODEL=saul-7b-instruct`.

The inference prompt is **graded**: it answers from the retrieved context and
cites Articles precisely, and only falls back to the model's own knowledge
when context is insufficient — clearly labeled `General knowledge (not in
retrieved text - verify):`. The **citation critic** then strips or rewrites any
Article numbers in the draft that do not appear in the retrieved passages.

## Conversational features

### Query rewrite (`QUERY_REWRITE=1`)

When chat history is present, a follow-up is rewritten into a standalone retrieval
query (e.g. "what are its exceptions?" → "What are the exceptions to Article 21?").
Single-shot questions skip the rewriter. The CLI and Streamlit UI show the rewritten
query when it differs from the user's message.

### Citation critic (`CRITIC_ENABLED=1`, `CRITIC_MAX_REWRITES=2`)

After the LLM draft, the critic checks every cited Article against the retrieved
passages. Up to two LLM rewrite attempts remove ungrounded citations; if citations
are still ungrounded, a safe fallback answer is returned (grounded excerpt + disclaimer).
Disable with `--no-critic` (CLI) or the sidebar toggle (UI).

### Streaming

The Streamlit UI runs retrieval, drafting, and the critic eagerly, then streams
the approved answer word-by-word. Retrieved passages and source Articles persist in
the chat for inspection.

## Results

**Retrieval quality** (`python -m backend.evaluate`, retrieval-only, deterministic,
no LLM) over the **31 answerable questions** — the 3 negative/no-gold questions are
excluded here (nothing to retrieve; they're scored by abstention below):

| config        | Hit@1 | Hit@3 | Hit@5 |   MRR | Recall@K |
|---------------|------:|------:|------:|------:|---------:|
| structured    |  0.74 |  0.97 |  0.97 | 0.844 |     0.97 |
| naive         |  0.71 |  0.87 |  0.94 | 0.803 |     0.97 |
| hybrid        |  0.74 |  0.94 |  0.97 | 0.833 |     1.00 |
| **hybrid+rr** | **0.94** | **1.00** | **1.00** | **0.968** | **1.00** |

The cross-encoder reranker lifts Hit@1 from 0.74 to **0.94** and MRR to **0.968**;
hybrid retrieval pushes Recall@K to **1.00** (the naive index backstops anything
the structured chunker mis-segments).

**Answer quality** (`--with-llm --judge qwen2.5:7b-instruct`) over all **34 questions**
(factual / scenario / negative):

| metric | score |
|--------|-------|
| citation hit rate | **0.968** |
| faithfulness (cited Articles present in retrieved context) | **0.912** |
| keyword recall | **0.909** |
| abstention on negative questions | **1.00** |
| LLM-judge correctness (1–5) | **4.09** |
| LLM-judge groundedness (1–5) | **4.62** |

> **Note:** These answer scores are from a **pre-critic** eval run. The production
> pipeline now includes the citation critic. Re-run
> `python -m backend.evaluate --with-llm` to measure post-critic impact (expect
> higher faithfulness; some answers may be thinner when retrieval misses).

Every `--with-llm` run writes a full transcript to `eval/runs/`. See
[Limitations](#limitations) for known edge cases.

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
│   ├── citations.py      # Article citation parsing / grounding checks
│   ├── critic.py         # post-generation citation faithfulness critic
│   ├── evaluate.py       # retrieval + answer eval harness
│   └── query.py          # answer() with rewrite, critic, graded prompt (+ CLI)
├── eval/
│   ├── questions.json    # labeled eval set (34 questions)
│   └── runs/             # JSON transcripts from --with-llm runs
├── ui/app.py             # Streamlit chat (hybrid / rerank / critic toggles)
├── scripts/
│   ├── Modelfile.saul    # Ollama Modelfile for SaulLM-7B
│   └── setup_saul.sh     # download GGUF + `ollama create`
├── requirements.txt
└── .env.example
```

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** running locally, with an inference model pulled:
  `ollama pull qwen2.5:7b-instruct`
- **poppler-utils** for `pdftotext` (e.g. `sudo apt install poppler-utils`).
  A `pypdf` fallback exists, but poppler gives the cleanest extraction.

## Setup

```bash
cd 05-labs/constitution-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

> The official Constitution PDF ships in `data/`, so no download is needed.
> First use downloads the bge embedding + reranker weights from Hugging Face
> (~440 MB + ~600 MB). Install a CUDA `torch` build to use your GPU.

**Optional — legal-tuned model.** To use SaulLM-7B instead of qwen, download the
GGUF and register it with Ollama, then point the app at it:

```bash
bash scripts/setup_saul.sh                 # creates the saul-7b-instruct model
export OLLAMA_MODEL=saul-7b-instruct
```

## Workflow

```bash
export PYTHONPATH=.

# 1. Inspect chunking before embedding (no models needed).
python -m backend.compare --sample
python -m backend.dump --article 21

# 2. Build both vector collections (uses bge-base embeddings).
python -m backend.ingest                 # --mode naive|structured|both, --force

# 3. Ask questions (hybrid + rerank + rewrite + critic by default).
python -m backend.query                  # multi-turn CLI; add --show for passages
python -m backend.query --no-critic      # disable citation critic
python -m backend.query --no-hybrid --strategy structured
python -m backend.query --no-rerank

# 4. Web UI (streaming chat, example prompts, sidebar toggles).
streamlit run ui/app.py

# 5. Measure retrieval quality across configs (structured/naive/hybrid/+rerank).
python -m backend.evaluate                # Hit@1/3/5, MRR, Recall@K
python -m backend.evaluate --k 10 --verbose

# 6. Answer-level eval (runs the full pipeline): citation/faithfulness/keyword/abstention.
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

**Answer eval** (`--with-llm`) runs the production pipeline (rewrite + critic when
enabled) and scores the generated answers:

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
- Chat: `QUERY_REWRITE`, `CHAT_HISTORY_TURNS`.
- Critic: `CRITIC_ENABLED`, `CRITIC_MAX_REWRITES`.
- LLM: `OLLAMA_MODEL`, `OLLAMA_HOST`.

## Limitations

- **English only.** The PDF's English text layer is used; the Hindi pages are
  scanned images and are not ingested (no OCR).
- **Known retrieval edge case.** For "*which body* recommends distribution of
  taxes" the embedder favours the tax-distribution Articles (269/270) over the
  Finance Commission (280), so that one answer can cite a neighbour. Tracked in
  the eval (it's the single non-clean case in the retrieval results above).
- **Critic fallback.** When rewrites fail to remove ungrounded citations, the
  critic returns a thin safe excerpt instead of a full explanatory answer.
- **Rewrite anchoring.** Multi-topic compare questions in a follow-up can be
  rewritten toward the wrong Article if the prior turn mentioned one (e.g.
  comparing emergency types → anchored to Article 359).
- **Chunking metadata.** Occasional mis-tagged Article numbers in structured
  chunks (known ingest edge case).
- **Self-judging caveat.** The judge scores above use the same model family that
  produced the answers; use a different `--judge` model for a stricter signal.
- Not legal advice — this is a retrieval/IR project over the constitutional text.

## Roadmap

- **MCP server (planned)** — expose constitution search (and optionally the full
  answer pipeline) as [Model Context Protocol](https://modelcontextprotocol.io) tools
  for Cursor, Claude Desktop, and other MCP hosts.

## Notes

- Both collections share one `chroma_db/` and use cosine space.
- Text extraction is cached to `data/constitution.txt`; use `--force` to rebuild.
- No OCR: the PDF has a real text layer (English). The Hindi pages are images.

## License

The **code** in this repository is released under the [MIT License](LICENSE).
The bundled **Constitution of India PDF** is an official publication of the
Government of India (Ministry of Law and Justice, Legislative Department) and is
included here only as source data for the RAG pipeline; rights to that text
remain with its publisher.
