"""
Scraper for Boston Housing Authority FAQ page.
Single page (depth 0), parse each Q&A pair as an individual document.
URL: https://www.bostonhousing.org/en/FAQ.aspx
"""

import re
from urllib.parse import urlparse

from .utils import (
    clean_html,
    extract_legal_citations,
    extract_section_headers,
    fetch_page,
    html_to_markdown,
    make_doc_id,
    parse_html,
    save_document,
    save_raw_html,
)

ROOT_URL = "https://www.bostonhousing.org/en/FAQ.aspx"
SOURCE_NAME = "bostonhousing"

# Additional pages beyond the FAQ
ADDITIONAL_PAGES = [
    {"url": "https://www.bostonhousing.org/en/For-Section-8-Participants.aspx", "label": "section_8_participants"},
    {"url": "https://www.bostonhousing.org/en/How-to-Apply.aspx", "label": "how_to_apply"},
    {"url": "https://www.bostonhousing.org/en/Grievance-Procedures.aspx", "label": "grievance_procedures"},
    {"url": "https://www.bostonhousing.org/en/For-Tenants.aspx", "label": "for_tenants"},
    {"url": "https://www.bostonhousing.org/en/Resident-Services.aspx", "label": "resident_services"},
]


def parse_faq_pairs(soup) -> list[dict]:
    """Parse Q&A pairs from the BHA FAQ page.

    The page structure varies, so we try multiple strategies:
    1. Look for FAQ-specific classes/IDs
    2. Look for accordion/toggle patterns
    3. Fall back to heading + content pairs
    """
    pairs = []

    # Strategy 1: Look for FAQ items with common class patterns
    faq_items = soup.find_all(class_=re.compile(r"faq|qa|question|accordion", re.I))
    if faq_items:
        for item in faq_items:
            q_el = item.find(class_=re.compile(r"question|title|header|toggle", re.I))
            a_el = item.find(class_=re.compile(r"answer|body|content|panel", re.I))
            if q_el and a_el:
                q = q_el.get_text(strip=True)
                a = a_el.get_text(strip=True)
                if q and a:
                    pairs.append({"question": q, "answer": a, "category": ""})
        if pairs:
            return pairs

    # Strategy 2: Look for dt/dd pairs (definition lists)
    dls = soup.find_all("dl")
    for dl in dls:
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            q = dt.get_text(strip=True)
            a = dd.get_text(strip=True)
            if q and a:
                pairs.append({"question": q, "answer": a, "category": ""})
    if pairs:
        return pairs

    # Strategy 3: h3/h4 as questions, following content as answers
    current_category = ""
    headings = soup.find_all(["h2", "h3", "h4"])
    for heading in headings:
        text = heading.get_text(strip=True)
        if not text:
            continue

        # If it's an h2, treat as category
        if heading.name == "h2":
            current_category = text
            continue

        # Collect all content until next heading
        answer_parts = []
        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in ["h2", "h3", "h4"]:
            part = sibling.get_text(strip=True)
            if part:
                answer_parts.append(part)
            sibling = sibling.find_next_sibling()

        answer = "\n".join(answer_parts)
        if text and answer:
            pairs.append({
                "question": text,
                "answer": answer,
                "category": current_category,
            })

    return pairs


def scrape_additional_page(url: str, label: str) -> dict | None:
    """Scrape a non-FAQ BHA page as a full guide document."""
    resp = fetch_page(url)
    if not resp:
        print(f"  [SKIP] Could not fetch {url}")
        return None

    save_raw_html(resp.text, SOURCE_NAME, f"{label}.html")

    soup = parse_html(resp.text)
    main = soup.find("main") or soup.find(id=re.compile(r"content|main", re.I)) or soup
    headers = extract_section_headers(main)
    cleaned = clean_html(main)
    markdown = html_to_markdown(cleaned)

    if len(markdown) < 100:
        print(f"  [SKIP] Too short: {url}")
        return None

    doc_id = make_doc_id(SOURCE_NAME, label)
    # Extract title from page
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else label.replace("_", " ").title()

    save_document(
        doc_id=doc_id,
        source_url=url,
        source_name=SOURCE_NAME,
        title=title,
        content=markdown,
        content_type="guide",
        crawl_depth=0,
        section_headers=headers,
    )
    return {"doc_id": doc_id, "url": url, "title": title}


def run():
    """Main entry point for BHA FAQ scraping."""
    print("=" * 60)
    print("Scraping BHA FAQ - bostonhousing.org")
    print("=" * 60)

    resp = fetch_page(ROOT_URL)
    if not resp:
        print("[ERROR] Could not fetch BHA FAQ page")
        return []

    save_raw_html(resp.text, SOURCE_NAME, "faq.html")

    soup = parse_html(resp.text)
    # Try to find the main content area
    main = soup.find("main") or soup.find(id=re.compile(r"content|main", re.I)) or soup
    cleaned = clean_html(main)

    faq_pairs = parse_faq_pairs(cleaned)
    print(f"  Found {len(faq_pairs)} Q&A pairs")

    all_docs = []
    for i, pair in enumerate(faq_pairs):
        q = pair["question"]
        a = pair["answer"]
        category = pair["category"]

        content = f"**Q: {q}**\n\n{a}"
        slug = re.sub(r"[^a-z0-9]+", "_", q.lower())[:60]
        doc_id = make_doc_id(SOURCE_NAME, f"faq_{i:03d}_{slug}")

        title = q
        if category:
            title = f"[{category}] {q}"

        save_document(
            doc_id=doc_id,
            source_url=ROOT_URL,
            source_name=SOURCE_NAME,
            title=title,
            content=content,
            content_type="faq",
            crawl_depth=0,
            section_headers=[category] if category else [],
        )
        all_docs.append({"doc_id": doc_id, "question": q})

    # Also save the full page as one document in case Q&A parsing misses things
    full_text = cleaned.get_text(separator="\n", strip=True)
    if len(full_text) > 200:
        save_document(
            doc_id=make_doc_id(SOURCE_NAME, "faq_full_page"),
            source_url=ROOT_URL,
            source_name=SOURCE_NAME,
            title="Boston Housing Authority - Complete FAQ",
            content=full_text,
            content_type="faq",
            crawl_depth=0,
        )

    # Scrape additional BHA pages
    print("\n--- Scraping additional BHA pages ---")
    for page in ADDITIONAL_PAGES:
        print(f"\n[Additional] {page['url']}")
        doc = scrape_additional_page(page["url"], page["label"])
        if doc:
            all_docs.append(doc)

    print(f"\n{'=' * 60}")
    print(f"Done! Scraped {len(all_docs)} documents from BHA")
    print(f"{'=' * 60}")
    return all_docs


if __name__ == "__main__":
    run()
