"""
Shared utilities for web scraping: fetching, cleaning, saving, citation extraction.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Polite crawling defaults
REQUEST_DELAY = 2  # seconds between requests
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 30

# Legal citation patterns for Massachusetts
CITATION_PATTERNS = [
    r"MGL\s+c\.?\s*\d+[A-Z]?\s*,?\s*(?:s\.|§|sec(?:tion)?\.?)\s*\d+[A-Z]?",  # MGL c.186, s.15B
    r"M\.?G\.?L\.?\s+c(?:h(?:apter)?)?\.?\s*\d+[A-Z]?\s*,?\s*(?:s\.|§|sec(?:tion)?\.?)\s*\d+[A-Z]?",
    r"G\.?L\.?\s+c\.?\s*\d+[A-Z]?\s*,?\s*§?\s*\d+[A-Z]?",
    r"Chapter\s+\d+[A-Z]?\s*,?\s*Section\s+\d+[A-Z]?",
    r"\d{3}\s+CMR\s+\d+\.\d+",  # 940 CMR 3.17
    r"\d{3}\s+C\.?M\.?R\.?\s+\d+\.\d+",
]


def fetch_page(url: str, delay: float = REQUEST_DELAY) -> requests.Response | None:
    """Fetch a page with polite delay and error handling."""
    time.sleep(delay)
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}")
        return None


def fetch_pdf(url: str, save_path: Path, delay: float = REQUEST_DELAY) -> Path | None:
    """Download a PDF file to save_path."""
    time.sleep(delay)
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Downloaded PDF: {save_path.name}")
        return save_path
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to download PDF {url}: {e}")
        return None


def parse_html(html: str) -> BeautifulSoup:
    """Parse HTML into BeautifulSoup."""
    return BeautifulSoup(html, "html.parser")


def clean_html(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove nav, footer, scripts, styles, cookie banners, and comments."""
    # Remove non-content elements
    for tag_name in ["nav", "footer", "header", "script", "style", "noscript", "iframe"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove common non-content classes/ids
    noise_patterns = [
        "cookie", "banner", "sidebar", "breadcrumb", "menu", "navigation",
        "social", "share", "related", "advertisement", "ad-", "footer",
        "skip-to", "back-to-top",
    ]
    for el in soup.find_all(True):
        if el.attrs is None:
            continue
        classes = " ".join(el.get("class", []))
        el_id = el.get("id", "")
        combined = f"{classes} {el_id}".lower()
        if any(p in combined for p in noise_patterns):
            el.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    return soup


def html_to_markdown(soup: BeautifulSoup) -> str:
    """Convert cleaned HTML to Markdown, preserving structure."""
    md = markdownify(str(soup), heading_style="ATX", strip=["img", "a"])
    # Normalize excessive whitespace but keep paragraph breaks
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"[ \t]+", " ", md)
    # Clean up lines
    lines = [line.strip() for line in md.split("\n")]
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def extract_legal_citations(text: str) -> list[str]:
    """Extract Massachusetts legal citations from text."""
    citations = set()
    for pattern in CITATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            citations.add(match.group(0).strip())
    return sorted(citations)


def extract_section_headers(soup: BeautifulSoup) -> list[str]:
    """Extract heading text from the page."""
    headers = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = tag.get_text(strip=True)
        if text:
            headers.append(text)
    return headers


def content_hash(text: str) -> str:
    """SHA-256 hash of content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_doc_id(source_name: str, slug: str) -> str:
    """Create a document ID from source name and slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    return f"{source_name}_{slug}"[:120]


def save_document(
    doc_id: str,
    source_url: str,
    source_name: str,
    title: str,
    content: str,
    content_type: str,
    crawl_depth: int = 0,
    section_headers: list[str] | None = None,
    parent_url: str | None = None,
) -> Path:
    """Save a processed document as JSON to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    doc = {
        "doc_id": doc_id,
        "source_url": source_url,
        "source_name": source_name,
        "title": title,
        "section_headers": section_headers or [],
        "content": content,
        "content_type": content_type,
        "crawl_depth": crawl_depth,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "legal_citations": extract_legal_citations(content),
        "parent_url": parent_url,
        "content_hash": content_hash(content),
    }

    filepath = PROCESSED_DIR / f"{doc_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {filepath.name} ({len(content)} chars)")
    return filepath


def save_raw_html(html: str, source_name: str, filename: str) -> Path:
    """Save raw HTML to data/raw/<source_name>/."""
    raw_dir = RAW_DIR / source_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    filepath = raw_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath


def extract_links(soup: BeautifulSoup, base_url: str, allowed_domains: list[str] | None = None) -> list[str]:
    """Extract and resolve links, optionally filtering by domain."""
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        # Remove fragments
        full_url = full_url.split("#")[0]
        # Filter by domain if specified
        if allowed_domains:
            domain = urlparse(full_url).netloc
            if not any(d in domain for d in allowed_domains):
                continue
        # Only HTTP(S) links
        if full_url.startswith("http"):
            links.add(full_url)
    return sorted(links)


def extract_pdf_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract PDF download links from a page (hrefs ending in .pdf or /download)."""
    pdf_links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        full_url = full_url.split("#")[0]
        if full_url.lower().endswith(".pdf") or "/download" in full_url.lower():
            if full_url.startswith("http"):
                pdf_links.add(full_url)
    return sorted(pdf_links)


def is_relevant_renter_url(url: str) -> bool:
    """Check if a URL is likely relevant to renter/tenant topics."""
    keywords = [
        "rent", "tenant", "evict", "lease", "landlord", "housing",
        "deposit", "voucher", "section-8", "habitability", "repair",
        "mold", "bed-bug", "discrimination", "fair-housing",
    ]
    url_lower = url.lower()
    return any(kw in url_lower for kw in keywords)


def load_processed_docs() -> list[dict]:
    """Load all processed JSON documents."""
    docs = []
    if not PROCESSED_DIR.exists():
        return docs
    for filepath in sorted(PROCESSED_DIR.glob("*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            docs.append(json.load(f))
    return docs
