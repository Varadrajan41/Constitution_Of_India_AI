"""Two chunking strategies over the same extracted text.

Both are *offset based*: every chunk records its [start, end) span in the source
string. This lets `compare.py` prove coverage (the union of spans) so we can be
certain no text is silently dropped by either strategy.

- naive_chunks:      fixed char windows with overlap over the whole document.
                     Coverage is 100% by construction.
- structured_chunks: partitions the document on Preamble / PART / Article /
                     SCHEDULE boundaries, tracking Part/Schedule context for
                     clean citations. It *partitions* the text (contiguous,
                     gap-free), so nothing is skipped even if a header is missed
                     - that text simply stays merged with the previous unit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend import config


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict)


# --- Naive -----------------------------------------------------------------
def naive_chunks(text: str, size: int = None, overlap: int = None) -> list[Chunk]:
    size = size or config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(text), step)):
        end = min(start + size, len(text))
        body = text[start:end]
        if body.strip():
            chunks.append(
                Chunk(
                    text=body,
                    start=start,
                    end=end,
                    metadata={"type": "naive", "window": i},
                )
            )
        if end == len(text):
            break
    return chunks


# --- Structured ------------------------------------------------------------
_PREAMBLE = re.compile(r"WE,\s+THE\s+PEOPLE\s+OF\s+INDIA")
# An Article/section header at line start: optional footnote digit + optional "[",
# then NN or NNA, a dot, space, then a capitalised title. The negative lookahead
# rejects footnote/amendment lines such as "1. Subs. by ..." / "2. Ins. by ...".
# Headings at the top of a page are preceded by a form-feed (\f), so leading
# whitespace classes include \f as well as spaces/tabs. A footnote superscript
# only appears as "<digits>[" (e.g. 2[21A), so we require the bracket there -
# otherwise the leading digits would steal part of a 3-digit article number
# (e.g. "368." being misread as "8"). Article suffixes can be up to two letters
# (e.g. 243ZG).
# Word boundaries matter: "Rep\b" must not reject the title "Repeals" (Art 395),
# but must reject the footnote "Rep.". An optional "<digits>[" may wrap the title
# itself (e.g. 368. 1[Power of Parliament ...]).
_NOTE_WORDS = (
    r"(?:Subs\b|Ins\b|Omitted\b|Added\b|Rep\b|Cl\.|Sub-|ibid\b|w\.e\.f|The words?\b)"
)
# Leading "junk" a header line may start with: spaces, tabs, form-feed, and
# footnote marker glyphs (asterisk, dagger, double-dagger, and Private-Use-Area
# symbols such as \uf02a that some article titles begin with, e.g. Art 370).
_LEAD = r"[ \t\f*\u2020\u2021\uf000-\uf0ff]*"
_ARTICLE = re.compile(
    r"(?m)^" + _LEAD + r"(?:\d+\[|\[)?\s*(\d{1,3}[A-Z]{0,2})\.\s+(?:\d+\[)?"
    r"(?!" + _NOTE_WORDS + r")(?=[A-Z\u201c\"])"
)
_PART = re.compile(r"(?m)^" + _LEAD + r"PART\s+([IVXLC]+)\b[ \t]*(.*)$")
_SCHEDULE = re.compile(
    r"(?m)^" + _LEAD + r"(?:\d+\[|\[)?\s*"
    r"(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH)\s+SCHEDULE\b"
)
# Footnote / amendment-note lines (kept in the chunk, flagged in metadata).
_NOTE = re.compile(r"(?m)^[ \t]*(?:\d+\.\s+(?:Subs|Ins|Omitted|Added|Rep|The words)|"
                   r"Cl\.|Sub-clause|Provided that nothing).*")


def _title_of(segment: str) -> str:
    s = re.sub(r"^" + _LEAD, "", segment.lstrip("\n"))
    m = re.match(
        r"(?:\d+\[|\[)?\s*\d{1,3}[A-Z]{0,2}\.\s+(?:\d+\[)?(.*?)[\u2014]", s, re.S
    )
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip().rstrip("].-\u2014").strip()


def _windows(start: int, end: int, text: str, size: int, overlap: int):
    """Yield contiguous [s, e) windows that together cover [start, end)."""
    step = max(1, size - overlap)
    s = start
    while s < end:
        e = min(s + size, end)
        yield s, e
        if e == end:
            break
        s += step


def structured_chunks(text: str) -> list[Chunk]:
    n = len(text)
    pre = _PREAMBLE.search(text)
    preamble_start = pre.start() if pre else 0

    # Collect boundaries only within the body (ignore the table of contents).
    boundaries: list[tuple[int, str, dict]] = [(0, "front_matter", {})]
    if pre:
        boundaries.append((preamble_start, "preamble", {}))

    for m in _PART.finditer(text):
        if m.start() >= preamble_start:
            label = f"Part {m.group(1)}"
            boundaries.append((m.start(), "part_heading", {"part": label}))
    for m in _SCHEDULE.finditer(text):
        if m.start() >= preamble_start:
            label = f"{m.group(1).title()} Schedule"
            boundaries.append((m.start(), "schedule_heading", {"schedule": label}))
    for m in _ARTICLE.finditer(text):
        if m.start() >= preamble_start:
            boundaries.append((m.start(), "article", {"number": m.group(1)}))

    # De-duplicate by offset (a line can match only one role); sort.
    seen = {}
    for off, kind, meta in boundaries:
        seen.setdefault(off, (off, kind, meta))
    ordered = [seen[k] for k in sorted(seen)]
    offsets = [o for o, _, _ in ordered] + [n]

    chunks: list[Chunk] = []
    current_part = ""
    current_schedule = ""
    for i, (off, kind, meta) in enumerate(ordered):
        seg_start, seg_end = off, offsets[i + 1]
        segment = text[seg_start:seg_end]

        if kind == "part_heading":
            current_part = meta.get("part", current_part)
            current_schedule = ""
        elif kind == "schedule_heading":
            current_schedule = meta.get("schedule", current_schedule)

        container = current_schedule or current_part or ""
        if kind == "article" and current_schedule:
            ctype = "schedule_item"
        else:
            ctype = kind

        base_meta = {
            "type": ctype,
            "part": current_part,
            "schedule": current_schedule,
            "container": container,
            "article": meta.get("number", ""),
        }
        if ctype in ("article", "schedule_item"):
            base_meta["title"] = _title_of(segment)
        if _NOTE.search(segment):
            base_meta["has_notes"] = True

        if not segment.strip():
            continue

        # Sub-split overlong units (a few Articles are very long).
        if len(segment) > config.MAX_ARTICLE_CHARS:
            for w, (s, e) in enumerate(
                _windows(seg_start, seg_end, text, config.MAX_ARTICLE_CHARS, config.CHUNK_OVERLAP)
            ):
                part = text[s:e]
                if not part.strip():
                    continue
                meta_w = dict(base_meta, window=w)
                chunks.append(Chunk(text=part, start=s, end=e, metadata=meta_w))
        else:
            chunks.append(Chunk(text=segment, start=seg_start, end=seg_end, metadata=base_meta))

    return chunks


def build(strategy: str, text: str) -> list[Chunk]:
    strategy = strategy.lower()
    if strategy == "naive":
        return naive_chunks(text)
    if strategy == "structured":
        return structured_chunks(text)
    raise ValueError(f"Unknown strategy '{strategy}'.")
