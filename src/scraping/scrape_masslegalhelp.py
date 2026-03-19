"""
Scraper for MassLegalHelp.org "Legal Tactics: Tenants' Rights in Massachusetts" (9th ed, Jan 2025).
Downloads all 18 chapter PDFs, extracts text, and saves as processed documents.
Falls back to HTML scraping if PDF download fails.
"""

import re
from urllib.parse import urljoin, urlparse

import pdfplumber

from .utils import (
    RAW_DIR,
    clean_html,
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

SOURCE_NAME = "masslegalhelp"
INDEX_URL = "https://www.masslegalhelp.org/housing-apartments-shelter/tenants-rights/chapter-pdfs-legal-tactics"

# Chapter titles for metadata (ordered 1-18)
CHAPTER_TITLES = [
    "Before You Move In",
    "Tenant Screening",
    "Security Deposits",
    "Kinds of Tenancies",
    "Rent",
    "Utilities",
    "Discrimination",
    "Getting Repairs Made",
    "Lead Poisoning",
    "Getting Organized",
    "Moving Out",
    "Evictions",
    "When to Take Your Landlord to Court",
    "Using the Court System",
    "Rooming Houses",
    "Mobile Homes",
    "Condominium Control",
    "Tenants and Foreclosure",
]


def scrape_chapter_pdf(pdf_url: str, chapter_num: int, chapter_title: str) -> dict | None:
    """Download a chapter PDF and extract its text."""
    filename = f"chapter_{chapter_num:02d}_{re.sub(r'[^a-z0-9]+', '_', chapter_title.lower()).strip('_')}.pdf"
    save_path = RAW_DIR / SOURCE_NAME / filename

    result = fetch_pdf(pdf_url, save_path)
    if not result:
        return None

    full_text = []
    try:
        with pdfplumber.open(save_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
    except Exception as e:
        print(f"  [ERROR] PDF extraction failed for chapter {chapter_num}: {e}")
        return None

    content = "\n\n".join(full_text)
    if len(content) < 100:
        print(f"  [SKIP] Chapter {chapter_num} PDF too short ({len(content)} chars)")
        return None

    doc_id = make_doc_id(SOURCE_NAME, f"ch{chapter_num:02d}_{chapter_title}")
    title = f"Legal Tactics Ch. {chapter_num}: {chapter_title}"

    save_document(
        doc_id=doc_id,
        source_url=pdf_url,
        source_name=SOURCE_NAME,
        title=title,
        content=content,
        content_type="guide",
        crawl_depth=0,
        parent_url=INDEX_URL,
    )
    return {"doc_id": doc_id, "url": pdf_url, "title": title}


def scrape_chapter_html(html_url: str, chapter_num: int, chapter_title: str) -> dict | None:
    """Fallback: scrape the HTML version of a chapter."""
    resp = fetch_page(html_url)
    if not resp:
        return None

    slug = urlparse(html_url).path.strip("/").replace("/", "_")
    save_raw_html(resp.text, SOURCE_NAME, f"{slug}.html")

    soup = parse_html(resp.text)
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    headers = extract_section_headers(main)
    cleaned = clean_html(main)
    markdown = html_to_markdown(cleaned)

    if len(markdown) < 100:
        print(f"  [SKIP] Chapter {chapter_num} HTML too short")
        return None

    doc_id = make_doc_id(SOURCE_NAME, f"ch{chapter_num:02d}_{chapter_title}")
    title = f"Legal Tactics Ch. {chapter_num}: {chapter_title}"

    save_document(
        doc_id=doc_id,
        source_url=html_url,
        source_name=SOURCE_NAME,
        title=title,
        content=markdown,
        content_type="guide",
        crawl_depth=0,
        section_headers=headers,
        parent_url=INDEX_URL,
    )
    return {"doc_id": doc_id, "url": html_url, "title": title}


def match_pdf_to_chapter(pdf_url: str, chapter_titles: list[str]) -> tuple[int, str] | None:
    """Try to match a PDF URL to a chapter number/title based on URL text."""
    url_lower = pdf_url.lower()
    # Try matching by chapter number pattern in URL
    num_match = re.search(r"chapter[_-]?(\d{1,2})", url_lower)
    if num_match:
        num = int(num_match.group(1))
        if 1 <= num <= len(chapter_titles):
            return num, chapter_titles[num - 1]
    # Try matching by keyword in URL
    for i, title in enumerate(chapter_titles, 1):
        keywords = re.sub(r"[^a-z0-9]+", " ", title.lower()).split()
        # Require at least 2 keyword matches for multi-word titles, 1 for single-word
        min_matches = min(2, len(keywords))
        if sum(1 for kw in keywords if kw in url_lower) >= min_matches:
            return i, title
    return None


def run():
    """Main entry point for masslegalhelp.org scraping."""
    print("=" * 60)
    print("Scraping MassLegalHelp - Legal Tactics (18 chapters)")
    print("=" * 60)

    all_docs = []

    # Step 1: Fetch the chapter index page to find PDF links
    print(f"\nFetching chapter index: {INDEX_URL}")
    resp = fetch_page(INDEX_URL)
    if not resp:
        print("[ERROR] Could not fetch chapter index page")
        return []

    save_raw_html(resp.text, SOURCE_NAME, "chapter_index.html")
    soup = parse_html(resp.text)
    pdf_links = extract_pdf_links(soup, INDEX_URL)
    print(f"  Found {len(pdf_links)} PDF links on index page")

    # Step 2: Match PDFs to chapters and download
    matched_chapters = set()
    for pdf_url in pdf_links:
        match = match_pdf_to_chapter(pdf_url, CHAPTER_TITLES)
        if match and match[0] not in matched_chapters:
            chapter_num, chapter_title = match
            print(f"\n[Chapter {chapter_num}] {chapter_title}")
            print(f"  PDF: {pdf_url}")
            doc = scrape_chapter_pdf(pdf_url, chapter_num, chapter_title)
            if doc:
                all_docs.append(doc)
                matched_chapters.add(chapter_num)

    # Step 3: For any chapters not found as PDFs, try HTML fallback
    missing_chapters = set(range(1, 19)) - matched_chapters
    if missing_chapters:
        print(f"\n--- Attempting HTML fallback for {len(missing_chapters)} chapters ---")
        # Try to find HTML chapter links from the index page
        from .utils import extract_links
        html_links = extract_links(soup, INDEX_URL, allowed_domains=["masslegalhelp.org"])

        for chapter_num in sorted(missing_chapters):
            chapter_title = CHAPTER_TITLES[chapter_num - 1]
            # Try to find an HTML link for this chapter
            found = False
            for link in html_links:
                match = match_pdf_to_chapter(link, CHAPTER_TITLES)
                if match and match[0] == chapter_num and not link.lower().endswith(".pdf"):
                    print(f"\n[Chapter {chapter_num} - HTML fallback] {chapter_title}")
                    print(f"  URL: {link}")
                    doc = scrape_chapter_html(link, chapter_num, chapter_title)
                    if doc:
                        all_docs.append(doc)
                        found = True
                        break
            if not found:
                print(f"  [MISS] Chapter {chapter_num}: {chapter_title} - no PDF or HTML found")

    print(f"\n{'=' * 60}")
    print(f"Done! Scraped {len(all_docs)} chapters from MassLegalHelp")
    print(f"{'=' * 60}")
    return all_docs


if __name__ == "__main__":
    run()
