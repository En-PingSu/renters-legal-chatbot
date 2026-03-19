"""
Scraper for Greater Boston Legal Services (GBLS) housing fact sheets.
GBLS publishes plain-language tenant guides on subletting, lease-breaking,
security deposit small claims, utility shutoffs, and more.
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
    make_doc_id,
    parse_html,
    save_document,
    save_raw_html,
)

SOURCE_NAME = "gbls"
ALLOWED_DOMAINS = ["gbls.org"]

ROOT_URLS = [
    "https://www.gbls.org/our-work/housing",
]

# Tenant-relevant keywords to filter fact sheets
RELEVANT_KEYWORDS = [
    "tenant", "renter", "landlord", "evict", "lease", "rent",
    "deposit", "housing", "utility", "heat", "repair", "habitability",
    "sublet", "discrimination", "section-8", "voucher", "mold",
    "bed-bug", "notice", "rights", "court", "small-claims",
]


def is_relevant_gbls_url(url: str) -> bool:
    """Check if a GBLS URL is likely about tenant/housing topics."""
    url_lower = url.lower()
    # Accept any housing-related URL
    if "housing" in url_lower:
        return True
    return any(kw in url_lower for kw in RELEVANT_KEYWORDS)


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
    """Extract main content from GBLS pages."""
    main = soup.find("main")
    if main:
        return main
    article = soup.find("article")
    if article:
        return article
    content = soup.find("div", class_=re.compile(r"content|field--body|entry-content"))
    if content:
        return content
    return soup.find("body") or soup


def scrape_page(url: str, depth: int, parent_url: str | None = None) -> tuple[dict | None, list[str]]:
    """Scrape a single GBLS page."""
    resp = fetch_page(url)
    if not resp:
        return None, []

    slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
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


def scrape_pdfs_from_page(url: str, soup, visited_pdfs: set) -> list[dict]:
    """Find and download relevant PDF links from a GBLS page."""
    pdf_links = extract_pdf_links(soup, url)
    docs = []
    for pdf_url in pdf_links:
        if pdf_url in visited_pdfs:
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


def run():
    """Main entry point for GBLS scraping."""
    print("=" * 60)
    print("Scraping GBLS - Greater Boston Legal Services Housing Resources")
    print("=" * 60)

    visited = set()
    visited_pdfs = set()
    all_docs = []

    for root_url in ROOT_URLS:
        if root_url in visited:
            continue
        visited.add(root_url)
        print(f"\n[Root] {root_url}")
        doc, child_links = scrape_page(root_url, depth=0)
        if doc:
            all_docs.append(doc)

        # Follow relevant child links at depth 1
        relevant_children = [
            l for l in child_links
            if l not in visited and is_relevant_gbls_url(l)
        ]
        print(f"  Following {len(relevant_children)} relevant child links")
        for link in relevant_children:
            if link in visited:
                continue
            visited.add(link)
            print(f"  [Depth 1] {link}")
            doc, grandchild_links = scrape_page(link, depth=1, parent_url=root_url)
            if doc:
                all_docs.append(doc)

    # Scan for PDFs in all visited pages
    print("\n--- Scanning for PDF documents ---")
    for url in list(visited):
        slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
        raw_path = RAW_DIR / SOURCE_NAME / f"{slug}.html"
        if raw_path.exists():
            with open(raw_path, "r", encoding="utf-8") as f:
                soup = parse_html(f.read())
            pdf_docs = scrape_pdfs_from_page(url, soup, visited_pdfs)
            all_docs.extend(pdf_docs)

    print(f"\n{'=' * 60}")
    print(f"Done! Scraped {len(all_docs)} documents from GBLS")
    print(f"{'=' * 60}")
    return all_docs


if __name__ == "__main__":
    run()
