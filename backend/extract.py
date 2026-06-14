"""PDF -> plain text extraction for the Constitution lab.

No OCR is required: the official PDF carries a real text layer. We use poppler's
`pdftotext` when available (best reading-order fidelity) and fall back to the
pure-Python `pypdf`. The result is cached to data/constitution.txt and reused by
both chunking strategies so extraction only happens once.

Page boundaries are preserved as form-feed (\\f) characters, which lets the
naive chunker optionally split per page.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from backend import config


def _via_pdftotext(pdf_path: Path) -> str | None:
    """Extract with poppler's pdftotext if the binary is installed."""
    if shutil.which("pdftotext") is None:
        return None
    # "-" writes to stdout; default layout keeps legal reading order best.
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="ignore")


def _via_pypdf(pdf_path: Path) -> str | None:
    """Pure-Python fallback. Joins pages with form feeds to mirror pdftotext."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(str(pdf_path))
    return "\f".join(page.extract_text() or "" for page in reader.pages)


def extract(pdf_path: Path = None, force: bool = False) -> str:
    """Return the document text, building (and caching) it if needed.

    If `pdf_path` is already a .txt file it is read directly.
    """
    pdf_path = Path(pdf_path or config.SOURCE_FILE)

    if pdf_path.suffix.lower() == ".txt":
        if not pdf_path.exists():
            sys.exit(f"[extract] Text file not found: {pdf_path}")
        return pdf_path.read_text(encoding="utf-8", errors="ignore")

    if not pdf_path.exists():
        sys.exit(
            f"[extract] Source PDF not found: {pdf_path}\n"
            f"          Place it in data/ or set CONSTITUTION_SOURCE."
        )

    if config.TEXT_CACHE.exists() and not force:
        print(f"[extract] Using cached text: {config.TEXT_CACHE}")
        return config.TEXT_CACHE.read_text(encoding="utf-8", errors="ignore")

    print(f"[extract] Extracting text from {pdf_path.name} ...")
    text = _via_pdftotext(pdf_path)
    engine = "pdftotext"
    if not text:
        text = _via_pypdf(pdf_path)
        engine = "pypdf"
    if not text:
        sys.exit(
            "[extract] No extractor available. Install poppler-utils "
            "(provides pdftotext) or `pip install pypdf`."
        )

    config.TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    config.TEXT_CACHE.write_text(text, encoding="utf-8")
    pages = text.count("\f") + 1
    print(
        f"[extract] {engine}: {len(text):,} chars across ~{pages} pages "
        f"-> cached at {config.TEXT_CACHE}"
    )
    return text


if __name__ == "__main__":
    force = "--force" in sys.argv
    extract(force=force)
