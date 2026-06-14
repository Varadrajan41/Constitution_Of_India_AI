"""Write the chunks to human-readable files for inspection BEFORE embedding.

    python -m backend.dump                 # both strategies -> preview/
    python -m backend.dump --mode structured
    python -m backend.dump --limit 20      # only first N chunks per strategy
    python -m backend.dump --article 21    # only structured chunks for an article

Each chunk is written with its metadata header and full text so you can scroll
through and judge the chunking quality. Files land in preview/.
"""
import argparse
from pathlib import Path

from backend import chunkers, config
from backend.extract import extract

PREVIEW_DIR = config.BASE_DIR / "preview"


def _format(chunk, i: int) -> str:
    m = chunk.metadata
    head_bits = [f"#{i}", m.get("type", "?")]
    if m.get("article"):
        head_bits.append(f"Article {m['article']}")
    if m.get("part"):
        head_bits.append(m["part"])
    if m.get("schedule"):
        head_bits.append(m["schedule"])
    if m.get("title"):
        head_bits.append(f'"{m["title"]}"')
    header = " | ".join(head_bits)
    span = f"chars [{chunk.start}:{chunk.end}] len={len(chunk.text)}"
    bar = "=" * 100
    return f"{bar}\n{header}\n{span}\n{'-' * 100}\n{chunk.text.strip()}\n"


def dump(strategy: str, text: str, limit: int = None, article: str = None) -> Path:
    chunks = chunkers.build(strategy, text)
    if article:
        chunks = [c for c in chunks if c.metadata.get("article") == article]
    if limit:
        chunks = chunks[:limit]

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = PREVIEW_DIR / f"{strategy}_chunks.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {strategy} chunks: {len(chunks)} shown\n\n")
        for i, c in enumerate(chunks):
            f.write(_format(c, i))
            f.write("\n")
    print(f"[dump] {len(chunks)} {strategy} chunks -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["naive", "structured", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks per strategy.")
    parser.add_argument("--article", default=None, help="Filter structured chunks by article number.")
    args = parser.parse_args()

    text = extract()
    strategies = ["naive", "structured"] if args.mode == "both" else [args.mode]
    for strategy in strategies:
        dump(strategy, text, limit=args.limit, article=args.article)


if __name__ == "__main__":
    main()
