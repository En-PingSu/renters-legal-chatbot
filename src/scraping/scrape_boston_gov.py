"""
Scraper for boston.gov renter-related pages.
Covers three sources:
  - boston.gov/renting-boston (depth 1)
  - boston.gov/departments/housing (depth 1, renter-relevant only)
  - boston.gov/help-housing (depth 1, renters sections only)
"""

import re
from urllib.parse import urlparse

import pdfplumber

from .utils import (
    RAW_DIR,
    clean_html,
    extract_links,
    extract_pdf_links,
    extract_section_headers,
    fetch_page,
    fetch_pdf,
    html_to_markdown,
    is_relevant_renter_url,
    make_doc_id,
    parse_html,
    save_document,
    save_raw_html,
)

SOURCE_NAME = "boston_gov"
ALLOWED_DOMAINS = ["boston.gov"]

SOURCES = [
    {
        "url": "https://www.boston.gov/renting-boston",
        "label": "renting-boston",
        "depth": 1,
        "filter_fn": None,  # scrape all child links
    },
    {
        "url": "https://www.boston.gov/departments/housing",
        "label": "departments-housing",
        "depth": 1,
        "filter_fn": is_relevant_renter_url,  # only renter-relevant
    },
    {
        "url": "https://www.boston.gov/help-housing",
        "label": "help-housing",
        "depth": 1,
        "filter_fn": is_relevant_renter_url,
    },
    {
        "url": "https://www.boston.gov/departments/housing/office-housing-stability/help-tenants-facing-eviction",
        "label": "eviction-help",
        "depth": 0,
        "filter_fn": None,
    },
    {
        "url": "https://www.boston.gov/housing/landlord-counseling",
        "label": "landlord-counseling",
        "depth": 0,
        "filter_fn": None,
    },
    {
        "url": "https://www.boston.gov/how-mayors-office-housing-can-help",
        "label": "mayors-office-housing",
        "depth": 1,
        "filter_fn": is_relevant_renter_url,
    },
    # Priority 6: ISD housing inspection & complaint info
    {
        "url": "https://www.boston.gov/departments/inspectional-services/inspecting-housing-boston",
        "label": "inspecting-housing",
        "depth": 0,
        "filter_fn": None,
    },
    {
        "url": "https://www.boston.gov/departments/inspectional-services/inspectional-services-constituent-services",
        "label": "isd-constituent-services",
        "depth": 0,
        "filter_fn": None,
    },
]


def title_from_soup(soup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "Untitled"


def get_main_content(soup):
    """Extract main content from boston.gov pages."""
    main = soup.find("main")
    if main:
        return main
    article = soup.find("article")
    if article:
        return article
    content = soup.find("div", class_=re.compile(r"node-content|body-content|field--body"))
    if content:
        return content
    return soup.find("body") or soup


def scrape_page(url: str, depth: int, label: str, parent_url: str | None = None) -> tuple[dict | None, list[str]]:
    """Scrape a single boston.gov page."""
    resp = fetch_page(url)
    if not resp:
        return None, []

    slug = urlparse(url).path.strip("/").replace("/", "_") or label
    save_raw_html(resp.text, SOURCE_NAME, f"{slug}.html")

    soup = parse_html(resp.text)
    title = title_from_soup(soup)
    main_content = get_main_content(soup)
    headers = extract_section_headers(main_content)

    child_links = extract_links(main_content, url, allowed_domains=ALLOWED_DOMAINS)

    cleaned = clean_html(main_content)
    markdown = html_to_markdown(cleaned)

    if len(markdown) < 50:
        print(f"  [SKIP] Too short: {url}")
        return None, child_links

    doc_id = make_doc_id(SOURCE_NAME, slug)

    save_document(
        doc_id=doc_id,
        source_url=url,
        source_name=SOURCE_NAME,
        title=title,
        content=markdown,
        content_type="guide",
        crawl_depth=depth,
        section_headers=headers,
        parent_url=parent_url,
    )

    return {"doc_id": doc_id, "url": url, "title": title}, child_links


def scrape_page_pdfs(url: str, soup, visited_pdfs: set) -> list[dict]:
    """Find and download relevant PDF links from a boston.gov page."""
    pdf_links = extract_pdf_links(soup, url)
    # Filter to housing/tenant-relevant PDFs
    relevant_keywords = [
        "eviction", "tenant", "landlord", "rent", "housing", "lease",
        "rights", "notice", "orientation", "guide",
    ]
    docs = []
    for pdf_url in pdf_links:
        if pdf_url in visited_pdfs:
            continue
        url_lower = pdf_url.lower()
        if not any(kw in url_lower for kw in relevant_keywords):
            continue
        visited_pdfs.add(pdf_url)
        slug = urlparse(pdf_url).path.strip("/").replace("/", "_")
        filename = re.sub(r"[^a-z0-9_.]", "_", slug.lower())[-80:] or "doc.pdf"
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        save_path = RAW_DIR / SOURCE_NAME / filename
        result = fetch_pdf(pdf_url, save_path)
        if not result:
            continue
        full_text = []
        try:
            with pdfplumber.open(save_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
        except Exception as e:
            print(f"  [ERROR] PDF extraction failed: {e}")
            continue
        content = "\n\n".join(full_text)
        if len(content) < 100:
            continue
        doc_id = make_doc_id(SOURCE_NAME, filename.replace(".pdf", ""))
        title = filename.replace(".pdf", "").replace("_", " ").title()
        save_document(
            doc_id=doc_id,
            source_url=pdf_url,
            source_name=SOURCE_NAME,
            title=title,
            content=content,
            content_type="guide",
            crawl_depth=0,
            parent_url=url,
        )
        docs.append({"doc_id": doc_id, "url": pdf_url, "title": title})
    return docs


def scrape_source(source: dict, visited: set) -> list[dict]:
    """Scrape a single boston.gov source and its child pages."""
    url = source["url"]
    label = source["label"]
    max_depth = source["depth"]
    filter_fn = source["filter_fn"]

    print(f"\n--- {label}: {url} ---")
    docs = []

    if url in visited:
        return docs
    visited.add(url)

    doc, child_links = scrape_page(url, depth=0, label=label)
    if doc:
        docs.append(doc)

    if max_depth >= 1:
        if filter_fn:
            child_links = [l for l in child_links if filter_fn(l)]
        print(f"  Following {len(child_links)} child links")

        for link in child_links:
            if link in visited:
                continue
            visited.add(link)
            print(f"  [Depth 1] {link}")
            doc, _ = scrape_page(link, depth=1, label=label, parent_url=url)
            if doc:
                docs.append(doc)

    return docs


def run():
    """Main entry point for boston.gov scraping."""
    print("=" * 60)
    print("Scraping boston.gov - Renter Resources")
    print("=" * 60)

    visited = set()
    visited_pdfs = set()
    all_docs = []

    for source in SOURCES:
        docs = scrape_source(source, visited)
        all_docs.extend(docs)

    # Scan all scraped pages for PDF links
    print("\n--- Scanning for PDF documents ---")
    for url in list(visited):
        resp = None
        # Re-fetch pages to find PDFs (use cached raw HTML if available)
        slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
        raw_path = RAW_DIR / SOURCE_NAME / f"{slug}.html"
        if raw_path.exists():
            with open(raw_path, "r", encoding="utf-8") as f:
                soup = parse_html(f.read())
            pdf_docs = scrape_page_pdfs(url, soup, visited_pdfs)
            all_docs.extend(pdf_docs)

    print(f"\n{'=' * 60}")
    print(f"Done! Scraped {len(all_docs)} documents from boston.gov")
    print(f"{'=' * 60}")
    return all_docs


if __name__ == "__main__":
    run()
