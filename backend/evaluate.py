"""Measure retrieval quality across configurations on a labeled question set.

Replaces "feels good / feels reliable" with numbers. For each config we compute,
over eval/questions.json:

  Hit@1 / Hit@3 / Hit@5  - fraction of questions whose expected Article appears
                           at rank <= k among retrieved chunks (type == article)
  MRR                    - mean reciprocal rank of the first correct Article
  Recall@K               - fraction with a correct Article anywhere in top-K

Configs compared:
  structured   - structured collection only, no rerank
  naive        - naive collection only, no rerank
  hybrid       - both collections merged by embedding distance, no rerank
  hybrid+rr    - both collections + cross-encoder rerank

Retrieval-only (no LLM) so it is fast and deterministic.

    python -m backend.evaluate                # full table
    python -m backend.evaluate --k 10 --verbose
"""
import argparse
import json
import re
import time

from backend import config

QUESTIONS_FILE = config.BASE_DIR / "eval" / "questions.json"

CONFIGS = [
    ("structured", {"mode": "single", "strategy": "structured", "rerank": False}),
    ("naive", {"mode": "single", "strategy": "naive", "rerank": False}),
    ("hybrid", {"mode": "hybrid", "rerank": False}),
    ("hybrid+rr", {"mode": "hybrid", "rerank": True}),
]


def _load_questions():
    if not QUESTIONS_FILE.exists():
        raise SystemExit(f"[evaluate] Missing question set: {QUESTIONS_FILE}")
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _query(col, qvec, k):
    res = col.query(
        query_embeddings=[qvec], n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def ranked_metas(question, cfg, k, client, ef):
    """Return an ordered list of metadata dicts for one config."""
    from backend.embeddings import embed_query
    from backend.rerank import rerank

    qvec = embed_query(question)

    if cfg["mode"] == "single":
        col = client.get_collection(
            name=config.collection_for(cfg["strategy"]), embedding_function=ef
        )
        return [m for _, m, _ in _query(col, qvec, k)]

    # hybrid: union both collections, dedupe by exact text
    seen, cands = set(), []
    for strat in ("naive", "structured"):
        col = client.get_collection(
            name=config.collection_for(strat), embedding_function=ef
        )
        for doc, meta, dist in _query(col, qvec, k):
            if doc.strip() in seen:
                continue
            seen.add(doc.strip())
            cands.append((doc, dict(meta), dist))

    if cfg["rerank"]:
        ranked = rerank(question, [(d, m) for d, m, _ in cands])
        return [m for _, m, _ in ranked]
    cands.sort(key=lambda x: x[2])  # ascending distance
    return [m for _, m, _ in cands]


def _build_article_index():
    """Map character ranges -> Article number using the structured article spans.

    This lets us score ANY chunk (including naive windows, which carry no article
    metadata) by *where it falls* in the document, so the strategies are judged
    on equal footing.
    """
    from backend.extract import extract
    from backend import chunkers

    spans = []
    for c in chunkers.structured_chunks(extract()):
        if c.metadata.get("type") == "article" and c.metadata.get("article"):
            spans.append((c.metadata["article"], c.start, c.end))
    return spans


def _articles_in(span, index):
    cs, ce = span
    if cs is None or ce is None:
        return set()
    return {num for num, s, e in index if min(ce, e) - max(cs, s) > 0}


def _first_hit_rank(metas, expected, index):
    expected = set(expected)
    for i, m in enumerate(metas):
        span = (m.get("char_start"), m.get("char_end"))
        if expected & _articles_in(span, index):
            return i + 1
    return None


def evaluate(k=10, verbose=False):
    from backend.embeddings import BGEEmbeddingFunction

    questions = _load_questions()
    client = _client()
    ef = BGEEmbeddingFunction()
    index = _build_article_index()  # position -> article map (strategy-agnostic)

    print(f"\nEvaluating {len(questions)} questions, K={k}\n")
    header = f"{'config':<14}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}{'Recall@K':>11}"
    print(header)
    print("-" * len(header))

    results = {}
    for name, cfg in CONFIGS:
        ranks = []
        for item in questions:
            metas = ranked_metas(item["q"], cfg, k, client, ef)
            rank = _first_hit_rank(metas, item["articles"], index)
            ranks.append(rank)
            if verbose:
                got = rank if rank else "miss"
                print(f"   [{name}] {item['articles']} -> rank {got} :: {item['q'][:50]}")
        results[name] = ranks

        n = len(ranks)
        hit1 = sum(1 for r in ranks if r and r <= 1) / n
        hit3 = sum(1 for r in ranks if r and r <= 3) / n
        hit5 = sum(1 for r in ranks if r and r <= 5) / n
        mrr = sum((1.0 / r) for r in ranks if r) / n
        recall = sum(1 for r in ranks if r) / n
        print(f"{name:<14}{hit1:>8.2f}{hit3:>8.2f}{hit5:>8.2f}{mrr:>8.3f}{recall:>11.2f}")

    return results


# --------------------------------------------------------------------------
# Answer-level evaluation (--with-llm): runs the production pipeline, scores the
# generated answers with deterministic metrics, stores full transcripts, and
# optionally adds an LLM-as-judge score.
# --------------------------------------------------------------------------
_CITE_RE = re.compile(
    r"[Aa]rticles?\s+((?:\d{1,3}[A-Z]{0,2})(?:\s*(?:,|and|&|to|/)\s*\d{1,3}[A-Z]{0,2})*)"
)
_ABSTAIN_MARKERS = (
    "general knowledge (not in retrieved",
    "not explicitly",
    "does not explicitly",
    "no specific article",
    "not a named",
    "not expressly",
    "is not mentioned",
    "not directly mentioned",
    "no explicit",
)

JUDGE_PROMPT = """You grade an answer about the Constitution of India. Be strict.

Question: {q}
Reference Article(s): {refs}
Retrieved context (the answer should rely on this):
{ctx}

Answer to grade:
{ans}

Respond with ONLY compact JSON, no prose:
{{"correctness": <1-5>, "grounded": <1-5>, "rationale": "<one short sentence>"}}"""


def _cited_articles(text):
    out = set()
    for m in _CITE_RE.finditer(text):
        out.update(re.findall(r"\d{1,3}[A-Z]{0,2}", m.group(1)))
    return out


def _retrieved_articles(passages, index):
    """Articles considered 'grounded': a passage's own article id (by position)
    AND any article numbers cross-referenced inside the passage text (e.g. Art 368
    enumerates 54/55/73; Art 359 mentions 20/21). Without the latter, legitimate
    in-text references look like hallucinations.
    """
    arts = set()
    for doc, meta, _ in passages:
        arts |= _articles_in((meta.get("char_start"), meta.get("char_end")), index)
        arts |= _cited_articles(doc)
    return arts


def _abstained(text):
    t = text.lower()
    return any(mk in t for mk in _ABSTAIN_MARKERS)


def _keyword_recall(text, keywords):
    if not keywords:
        return None
    t = text.lower()
    hit = sum(1 for kw in keywords if kw.lower() in t)
    return hit / len(keywords)


def _judge(client, judge_model, item, passages, answer_text):
    ctx = "\n".join(d[:300] for d, _, _ in passages)[:2500]
    prompt = JUDGE_PROMPT.format(
        q=item["q"], refs=", ".join(item.get("articles") or ["(none)"]),
        ctx=ctx, ans=answer_text,
    )
    try:
        resp = client.chat(model=judge_model,
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {"raw": raw}
    except Exception as exc:
        return {"error": str(exc)}


def evaluate_answers(k=None, judge_model=None):
    """Run the production pipeline per question, score answers, save a transcript."""
    import ollama

    from backend import query as rag

    questions = _load_questions()
    index = _build_article_index()
    client = ollama.Client(host=config.OLLAMA_HOST)

    items_out = []
    fact, neg = [], []
    print(f"\nAnswer eval: {len(questions)} questions | model={config.OLLAMA_MODEL}"
          f"{' | judge=' + judge_model if judge_model else ''}\n")

    for item in questions:
        try:
            answer_text, passages = rag.answer(item["q"])
        except RuntimeError as exc:
            raise SystemExit(str(exc))

        expected = set(item.get("articles") or [])
        cited = _cited_articles(answer_text)
        retrieved_arts = _retrieved_articles(passages, index)
        kind = item.get("kind", "factual")

        metrics = {
            "citation_hit": bool(expected & cited) if expected else None,
            "hallucinated": sorted(cited - retrieved_arts),
            "faithful": cited.issubset(retrieved_arts) if cited else True,
            "keyword_recall": _keyword_recall(answer_text, item.get("keywords")),
            "abstained": _abstained(answer_text),
        }
        rec = {
            "q": item["q"], "kind": kind, "expected_articles": sorted(expected),
            "cited_articles": sorted(cited), "answer": answer_text,
            "retrieved": [
                {"strategy": m.get("strategy"), "type": m.get("type"),
                 "article": m.get("article"), "score": round(s, 3),
                 "preview": d[:160].strip()}
                for d, m, s in passages
            ],
            "metrics": metrics,
        }
        if judge_model:
            rec["judge"] = _judge(client, judge_model, item, passages, answer_text)
        items_out.append(rec)

        (neg if kind == "negative" else fact).append(rec)
        flag = ("ABSTAIN ok" if (kind == "negative" and metrics["abstained"])
                else ("hit" if metrics["citation_hit"] else "MISS"))
        if metrics["hallucinated"]:
            flag += f" !halluc {metrics['hallucinated']}"
        print(f"  [{kind:<8}] {flag:<14} {item['q'][:54]}")

    # Aggregate.
    def _rate(rows, key):
        vals = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else 0.0

    kw_vals = [r["metrics"]["keyword_recall"] for r in fact
               if r["metrics"]["keyword_recall"] is not None]
    summary = {
        "n_total": len(questions), "n_factual": len(fact), "n_negative": len(neg),
        "citation_hit_rate": round(_rate(fact, "citation_hit"), 3),
        "faithfulness_rate": round(_rate(items_out, "faithful"), 3),
        "keyword_recall_mean": round(sum(kw_vals) / len(kw_vals), 3) if kw_vals else None,
        "abstention_rate_negatives": round(_rate(neg, "abstained"), 3) if neg else None,
        "hallucinated_count": sum(1 for r in items_out if r["metrics"]["hallucinated"]),
    }
    if judge_model:
        cs = [r["judge"].get("correctness") for r in items_out
              if isinstance(r.get("judge"), dict) and isinstance(r["judge"].get("correctness"), (int, float))]
        gs = [r["judge"].get("grounded") for r in items_out
              if isinstance(r.get("judge"), dict) and isinstance(r["judge"].get("grounded"), (int, float))]
        summary["judge_correctness_mean"] = round(sum(cs) / len(cs), 2) if cs else None
        summary["judge_grounded_mean"] = round(sum(gs) / len(gs), 2) if gs else None

    print("\nSummary:")
    for k_, v in summary.items():
        print(f"  {k_:<28} {v}")

    runs_dir = config.BASE_DIR / "eval" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(
        {"model": config.OLLAMA_MODEL, "judge": judge_model, "summary": summary,
         "items": items_out}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTranscript saved: {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Retrieve depth for metrics.")
    parser.add_argument("--verbose", action="store_true", help="Per-question ranks.")
    parser.add_argument("--with-llm", action="store_true",
                        help="Also generate + score answers (slower).")
    parser.add_argument("--judge", metavar="OLLAMA_MODEL", default=None,
                        help="Optional LLM-as-judge model (e.g. qwen2.5:7b-instruct).")
    args = parser.parse_args()

    if args.with_llm:
        evaluate_answers(k=args.k, judge_model=args.judge)
    else:
        evaluate(k=args.k, verbose=args.verbose)


if __name__ == "__main__":
    main()
