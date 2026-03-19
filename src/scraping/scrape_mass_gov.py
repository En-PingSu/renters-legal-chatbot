"""
Scraper for mass.gov Massachusetts landlord/tenant law pages.
Crawl depth: 2 levels from root URL.
Also follows links to malegislature.gov for MGL chapters.
Downloads the "Legal Tactics" PDF.
"""

import re
from urllib.parse import urlparse

import pdfplumber

from .utils import (
    PROJECT_ROOT,
    RAW_DIR,
    clean_html,
    extract_links,
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

ROOT_URL = "https://www.mass.gov/info-details/massachusetts-law-about-landlord-and-tenant"
ALLOWED_DOMAINS = ["mass.gov", "malegislature.gov"]
SKIP_DOMAINS = ["nolo.com", "masslandlords.net"]
SOURCE_NAME = "mass_gov"

# Additional root URLs to crawl at depth 1 (coverage gaps)
ADDITIONAL_ROOT_URLS = [
    "https://www.mass.gov/landlords-and-tenants-rights-and-responsibilities",
    "https://www.mass.gov/rental-toolkit-for-landlords-and-tenants",
    "https://www.mass.gov/security-deposits",
    "https://www.mass.gov/info-details/security-deposits-and-last-months-rent",
    "https://www.mass.gov/info-details/find-out-what-landlords-can-use-security-deposits-for",
    "https://www.mass.gov/info-details/learn-about-returning-or-getting-back-a-security-deposit",
    "https://www.mass.gov/info-details/housing-assistance-for-massachusetts-residents",
    "https://www.mass.gov/law-library/massachusetts-law-about-housing-and-real-estate",
]

# Direct statute URLs to scrape individually (Priority 1: student-relevant gaps)
STATUTE_URLS = [
    # Late fee restrictions
    "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186/Section15A",
    # Move-in fee prohibition (first, last, security, lock change only)
    "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186/Section15E",
    # Utility metering requirements
    "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186/Section15D",
    # DV/sexual assault lease termination (sections 19-23)
    "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186/Section19",
    "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186/Section20",
    "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186/Section21",
    "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186/Section22",
    "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186/Section23",
    # Stay of execution after eviction judgment
    "https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239/Section9",
    # Appeal of eviction judgment
    "https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239/Section10",
    # Bond requirements on appeal
    "https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239/Section12",
    # Consumer protection / treble damages (behind 940 CMR 3.17)
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXV/Chapter93A/Section2",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXV/Chapter93A/Section9",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXV/Chapter93A/Section11",
    # Board of Health inspection authority and enforcement (sections 127A-127L)
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127A",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127B",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127C",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127D",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127E",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127F",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127G",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127H",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127I",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127J",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127K",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXVI/Chapter111/Section127L",
]

# Priority 2: Housing Court self-help & summary process resources
COURT_RESOURCE_URLS = [
    "https://www.mass.gov/orgs/housing-court",
    "https://www.mass.gov/lists/court-forms-for-eviction",
    "https://www.mass.gov/how-to/file-a-small-claim-in-the-boston-municipal-court-district-court-or-housing-court",
    "https://www.mass.gov/info-details/tenants-guide-to-eviction",
    "https://www.mass.gov/info-details/respond-to-an-eviction-against-you",
]

# Priority 4: MCAD housing discrimination complaint process
MCAD_URLS = [
    "https://www.mass.gov/mcad-complaints-of-discrimination",
    "https://www.mass.gov/info-details/overview-of-housing-discrimination",
    "https://www.mass.gov/how-to/how-to-file-a-complaint-of-discrimination",
    "https://www.mass.gov/info-details/guide-to-the-mcad-case-process",
]

# Priority 5: DHCD / state subsidized housing programs
DHCD_URLS = [
    "https://www.mass.gov/how-to/apply-for-the-massachusetts-rental-voucher-program-mrvp",
    "https://www.mass.gov/rental-assistance-housing-vouchers",
]

# AG Guide to Landlord/Tenant Rights PDF (2025 edition)
AG_GUIDE_PDF_URL = "https://www.mass.gov/doc/2025-guide-to-landlord-tenant-rights-11182025/download"


def classify_content(url: str, text: str) -> str:
    """Classify content type based on URL and content."""
    url_lower = url.lower()
    if "malegislature.gov" in url_lower or "general-laws" in url_lower:
        return "statute"
    if "regulation" in url_lower or "cmr" in url_lower:
        return "regulation"
    if "faq" in url_lower:
        return "faq"
    if "policy" in url_lower or "executive-order" in url_lower:
        return "policy"
    return "guide"


def title_from_soup(soup) -> str:
    """Extract page title."""
    # Try og:title first, then <title>, then first h1
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
    """Extract the main content area of a mass.gov page."""
    # mass.gov uses <main> or specific content divs
    main = soup.find("main")
    if main:
        return main
    content = soup.find("div", class_=re.compile(r"field--name-body|page-content|main-content"))
    if content:
        return content
    return soup.find("body") or soup


def scrape_page(url: str, depth: int, parent_url: str | None = None) -> tuple[dict | None, list[str]]:
    """Scrape a single page. Returns (document_dict, child_links)."""
    resp = fetch_page(url)
    if not resp:
        return None, []

    # Save raw HTML
    slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
    save_raw_html(resp.text, SOURCE_NAME, f"{slug}.html")

    soup = parse_html(resp.text)
    title = title_from_soup(soup)
    main_content = get_main_content(soup)
    headers = extract_section_headers(main_content)

    # Get links before cleaning
    child_links = extract_links(main_content, url, allowed_domains=ALLOWED_DOMAINS)
    # Filter out skip domains
    child_links = [
        link for link in child_links
        if not any(d in urlparse(link).netloc for d in SKIP_DOMAINS)
    ]

    # Clean and convert
    cleaned = clean_html(main_content)
    markdown = html_to_markdown(cleaned)

    if len(markdown) < 50:
        print(f"  [SKIP] Too short after cleaning: {url}")
        return None, child_links

    content_type = classify_content(url, markdown)
    doc_id = make_doc_id(SOURCE_NAME, slug)

    filepath = save_document(
        doc_id=doc_id,
        source_url=url,
        source_name=SOURCE_NAME,
        title=title,
        content=markdown,
        content_type=content_type,
        crawl_depth=depth,
        section_headers=headers,
        parent_url=parent_url,
    )

    return {"doc_id": doc_id, "url": url, "title": title}, child_links


def scrape_pdf(url: str, filename: str, title: str | None = None) -> dict | None:
    """Download and extract text from a PDF."""
    save_path = RAW_DIR / SOURCE_NAME / filename
    result = fetch_pdf(url, save_path)
    if not result:
        return None

    # Extract text with pdfplumber
    full_text = []
    try:
        with pdfplumber.open(save_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
    except Exception as e:
        print(f"  [ERROR] PDF extraction failed for {filename}: {e}")
        return None

    content = "\n\n".join(full_text)
    if len(content) < 100:
        print(f"  [SKIP] PDF too short: {filename}")
        return None

    doc_id = make_doc_id(SOURCE_NAME, filename.replace(".pdf", ""))
    doc_title = title or filename.replace(".pdf", "").replace("_", " ").title()
    save_document(
        doc_id=doc_id,
        source_url=url,
        source_name=SOURCE_NAME,
        title=doc_title,
        content=content,
        content_type="guide",
        crawl_depth=0,
        parent_url=ROOT_URL,
    )
    return {"doc_id": doc_id, "url": url, "title": doc_title}


def run():
    """Main scraping entry point for mass.gov."""
    print("=" * 60)
    print("Scraping mass.gov - Massachusetts Landlord & Tenant Law")
    print("=" * 60)

    visited = set()
    all_docs = []

    # Level 0: Root page
    print(f"\n[Depth 0] {ROOT_URL}")
    doc, level1_links = scrape_page(ROOT_URL, depth=0)
    if doc:
        all_docs.append(doc)
    visited.add(ROOT_URL)

    # Level 1
    print(f"\n--- Level 1: {len(level1_links)} links found ---")
    level2_links = []
    for link in level1_links:
        if link in visited:
            continue
        visited.add(link)
        print(f"\n[Depth 1] {link}")
        doc, children = scrape_page(link, depth=1, parent_url=ROOT_URL)
        if doc:
            all_docs.append(doc)
        level2_links.extend(children)

    # Level 2
    level2_links = [l for l in set(level2_links) if l not in visited]
    print(f"\n--- Level 2: {len(level2_links)} links found ---")
    for link in level2_links:
        if link in visited:
            continue
        visited.add(link)
        print(f"\n[Depth 2] {link}")
        doc, _ = scrape_page(link, depth=2, parent_url=None)
        if doc:
            all_docs.append(doc)

    # Additional root URLs (depth 1 crawl each)
    print(f"\n--- Additional root URLs: {len(ADDITIONAL_ROOT_URLS)} ---")
    for root_url in ADDITIONAL_ROOT_URLS:
        if root_url in visited:
            continue
        visited.add(root_url)
        print(f"\n[Additional root] {root_url}")
        doc, child_links = scrape_page(root_url, depth=0)
        if doc:
            all_docs.append(doc)
        # Follow child links at depth 1, filtered for relevance
        relevant_children = [
            l for l in child_links
            if l not in visited and is_relevant_renter_url(l)
        ]
        for link in relevant_children:
            if link in visited:
                continue
            visited.add(link)
            print(f"  [Depth 1] {link}")
            doc, _ = scrape_page(link, depth=1, parent_url=root_url)
            if doc:
                all_docs.append(doc)

    # Direct statute URLs (Priority 1)
    print(f"\n--- Direct statute URLs: {len(STATUTE_URLS)} ---")
    for url in STATUTE_URLS:
        if url in visited:
            continue
        visited.add(url)
        print(f"\n[Statute] {url}")
        doc, _ = scrape_page(url, depth=0)
        if doc:
            all_docs.append(doc)

    # Court resource URLs (Priority 2)
    print(f"\n--- Court resource URLs: {len(COURT_RESOURCE_URLS)} ---")
    for url in COURT_RESOURCE_URLS:
        if url in visited:
            continue
        visited.add(url)
        print(f"\n[Court] {url}")
        doc, child_links = scrape_page(url, depth=0)
        if doc:
            all_docs.append(doc)
        # Follow child links at depth 1 for court resources
        relevant_children = [
            l for l in child_links
            if l not in visited and is_relevant_renter_url(l)
        ]
        for link in relevant_children:
            if link in visited:
                continue
            visited.add(link)
            print(f"  [Depth 1] {link}")
            doc, _ = scrape_page(link, depth=1, parent_url=url)
            if doc:
                all_docs.append(doc)

    # MCAD URLs (Priority 4)
    print(f"\n--- MCAD URLs: {len(MCAD_URLS)} ---")
    for url in MCAD_URLS:
        if url in visited:
            continue
        visited.add(url)
        print(f"\n[MCAD] {url}")
        doc, child_links = scrape_page(url, depth=0)
        if doc:
            all_docs.append(doc)
        relevant_children = [
            l for l in child_links
            if l not in visited and is_relevant_renter_url(l)
        ]
        for link in relevant_children:
            if link in visited:
                continue
            visited.add(link)
            print(f"  [Depth 1] {link}")
            doc, _ = scrape_page(link, depth=1, parent_url=url)
            if doc:
                all_docs.append(doc)

    # DHCD URLs (Priority 5)
    print(f"\n--- DHCD URLs: {len(DHCD_URLS)} ---")
    for url in DHCD_URLS:
        if url in visited:
            continue
        visited.add(url)
        print(f"\n[DHCD] {url}")
        doc, child_links = scrape_page(url, depth=0)
        if doc:
            all_docs.append(doc)
        relevant_children = [
            l for l in child_links
            if l not in visited and is_relevant_renter_url(l)
        ]
        for link in relevant_children:
            if link in visited:
                continue
            visited.add(link)
            print(f"  [Depth 1] {link}")
            doc, _ = scrape_page(link, depth=1, parent_url=url)
            if doc:
                all_docs.append(doc)

    # PDF: AG Guide to Landlord/Tenant Rights (2025)
    print("\n--- Downloading AG Guide PDF ---")
    pdf_doc = scrape_pdf(
        AG_GUIDE_PDF_URL,
        "ag_guide_landlord_tenant_2025.pdf",
        title="AG Guide to Landlord/Tenant Rights (2025)",
    )
    if pdf_doc:
        all_docs.append(pdf_doc)

    print(f"\n{'=' * 60}")
    print(f"Done! Scraped {len(all_docs)} documents from mass.gov")
    print(f"{'=' * 60}")
    return all_docs


if __name__ == "__main__":
    run()
