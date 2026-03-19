"""
Save mass.gov content that was fetched via browser (since requests gets 403).
This script processes pre-saved text files from browser extraction.
"""

import json
import re
from pathlib import Path

from .utils import (
    PROJECT_ROOT,
    RAW_DIR,
    extract_legal_citations,
    make_doc_id,
    save_document,
)

SOURCE_NAME = "mass_gov"
BROWSER_TEXT_DIR = RAW_DIR / SOURCE_NAME / "browser_text"


def save_browser_page(
    url: str,
    title: str,
    text: str,
    content_type: str = "guide",
    crawl_depth: int = 0,
    parent_url: str | None = None,
) -> dict | None:
    """Save a page that was fetched via browser."""
    if len(text) < 50:
        print(f"  [SKIP] Too short: {url}")
        return None

    slug = re.sub(r"[^a-z0-9]+", "_", url.split("//")[1])[:100]
    doc_id = make_doc_id(SOURCE_NAME, slug)

    # Extract section headers from markdown-style headers
    headers = re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)
    if not headers:
        # Try extracting from ALL-CAPS lines or bold text
        headers = re.findall(r"\*\*(.+?)\*\*", text)

    save_document(
        doc_id=doc_id,
        source_url=url,
        source_name=SOURCE_NAME,
        title=title,
        content=text,
        content_type=content_type,
        crawl_depth=crawl_depth,
        section_headers=headers[:20],
        parent_url=parent_url,
    )
    return {"doc_id": doc_id, "url": url, "title": title}


def save_from_text_file(filepath: Path, url: str, title: str, **kwargs) -> dict | None:
    """Load text from a file and save as document."""
    text = filepath.read_text(encoding="utf-8")
    return save_browser_page(url, title, text, **kwargs)
