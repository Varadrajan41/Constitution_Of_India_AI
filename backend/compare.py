"""Compare the two chunking strategies on the SAME extracted text.

Proves coverage (so we know no text is skipped) and prints side-by-side stats:
chunk counts, sizes, % of the document covered, and any uncovered gaps.

    python -m backend.compare            # summary table + coverage proof
    python -m backend.compare --sample   # also print a few example chunks
"""
import argparse
import statistics

from backend import chunkers
from backend.extract import extract


def _coverage(chunks, total):
    """Return (covered_chars, gaps) from the union of [start, end) spans."""
    spans = sorted((c.start, c.end) for c in chunks)
    covered, gaps, last = 0, [], 0
    # Merge intervals to count unique coverage and find holes.
    cur_s, cur_e = None, None
    union = []
    for s, e in spans:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            union.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        union.append((cur_s, cur_e))

    for s, e in union:
        covered += e - s
        if s > last:
            gaps.append((last, s))
        last = e
    if last < total:
        gaps.append((last, total))
    return covered, gaps


def _stats(name, chunks, total):
    sizes = [len(c.text) for c in chunks]
    covered, gaps = _coverage(chunks, total)
    return {
        "name": name,
        "chunks": len(chunks),
        "min": min(sizes) if sizes else 0,
        "avg": round(statistics.mean(sizes)) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "covered_pct": round(100 * covered / total, 3) if total else 0,
        "gaps": gaps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Print example chunks.")
    args = parser.parse_args()

    text = extract()
    total = len(text)

    naive = chunkers.naive_chunks(text)
    structured = chunkers.structured_chunks(text)

    rows = [_stats("naive", naive, total), _stats("structured", structured, total)]

    print(f"\nSource text: {total:,} chars\n")
    header = f"{'strategy':<12}{'chunks':>8}{'min':>8}{'avg':>8}{'max':>9}{'coverage':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<12}{r['chunks']:>8}{r['min']:>8}{r['avg']:>8}"
              f"{r['max']:>9}{r['covered_pct']:>10}%")

    # Coverage proof: report any uncovered gaps.
    print("\nCoverage check (no-info-skipped proof):")
    for r in rows:
        if not r["gaps"]:
            print(f"  {r['name']:<12} OK - 100% of characters are inside some chunk")
        else:
            shown = ", ".join(f"[{a}:{b}] ({b-a} chars)" for a, b in r["gaps"][:5])
            print(f"  {r['name']:<12} {len(r['gaps'])} gap(s): {shown}")

    # Structured-only breakdown.
    by_type = {}
    articles = set()
    for c in structured:
        by_type[c.metadata.get("type", "?")] = by_type.get(c.metadata.get("type", "?"), 0) + 1
        if c.metadata.get("type") == "article" and c.metadata.get("article"):
            articles.add(c.metadata["article"])
    print("\nStructured chunk types:")
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:<18}{n:>6}")
    print(f"  distinct article numbers detected: {len(articles)}")

    if args.sample:
        print("\n--- sample structured chunks ---")
        for c in [c for c in structured if c.metadata.get("type") == "article"][:3]:
            m = c.metadata
            print(f"\n[{m.get('part')}] Article {m.get('article')} - {m.get('title')}")
            print(c.text[:300].strip(), "...")


if __name__ == "__main__":
    main()
