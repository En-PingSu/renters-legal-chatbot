"""
Post-processing cleanup for chunked corpus.
Removes junk chunks (duplicates, headings-only, boilerplate, short fragments,
non-English content, nav menus, endnote-only chunks) to improve retrieval quality.
"""

import hashlib
import json
import re
from collections import defaultdict

from src.scraping.utils import PROJECT_ROOT

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"
MIN_CHUNK_LENGTH = 100  # raised from 50 to filter PDF extraction fragments


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks: list[dict]):
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Document-level filters (remove entire docs)
# ---------------------------------------------------------------------------

def remove_faq_full_page(chunks: list[dict]) -> list[dict]:
    """Remove chunks from the duplicate full-page FAQ document."""
    before = len(chunks)
    chunks = [c for c in chunks if c["doc_id"] != "bostonhousing_faq_full_page"]
    print(f"  Remove FAQ full page: {before} -> {len(chunks)} (-{before - len(chunks)})")
    return chunks


def remove_non_english_docs(chunks: list[dict]) -> list[dict]:
    """Remove chunks from non-English PDF documents (multilingual translations)."""
    non_english_keywords = [
        "arabic", "cape_verdean", "chinese", "haitian", "spanish",
        "vietnamese", "french", "portuguese", "russian", "somali",
    ]
    before = len(chunks)
    chunks = [
        c for c in chunks
        if not any(kw in c["doc_id"].lower() for kw in non_english_keywords)
    ]
    print(f"  Remove non-English docs: {before} -> {len(chunks)} (-{before - len(chunks)})")
    return chunks


def remove_oversized_reports(chunks: list[dict]) -> list[dict]:
    """Remove chunks from very large report PDFs that dilute the corpus."""
    oversized_docs = [
        "city_of_boston_assessment_of_fair_housing",
        "income_restricted_20housing",
        "boston_20housing_20report_202025",
        "neighborhood_20housing_20trust",
        "babel_20notice_20template",
    ]
    before = len(chunks)
    chunks = [
        c for c in chunks
        if not any(kw in c["doc_id"].lower() for kw in oversized_docs)
    ]
    print(f"  Remove oversized reports: {before} -> {len(chunks)} (-{before - len(chunks)})")
    return chunks


def remove_off_topic_docs(chunks: list[dict]) -> list[dict]:
    """Remove documents that are not directly about tenant/renter rights."""
    off_topic_docs = {
        "boston_gov_311",
        "boston_gov_departments_boston_311",
        "boston_gov_getting_around_boston",
        "boston_gov_buying_and_owning_home",
        "boston_gov_departments_housing_housing_initiatives_under_mayor_michelle_wu",
        "boston_gov_departments_housing_boston_home_center",
        "boston_gov_departments_housing_boston_home_center_boston_home_center_classes",
        "boston_gov_departments_housing_find_financial_help_owning_home",
        "boston_gov_departments_housing_how_apply_senior_home_repair",
        "boston_gov_departments_housing_annual_homeless_census",
        "boston_gov_departments_housing_real_estate_management_and_sales",
        "boston_gov_departments_housing_boston_acquisition_fund",
        "boston_gov_departments_housing_acquisition_opportunity_program",
        "boston_gov_moving",
        "boston_gov_buildinghousing",
        "boston_gov_default_files_file_2024_02_final_boston_20housing_20strategy_202025_20_2_0",
        "boston_gov_sites_default_files_file_2026_02_first_20term_20housing_20report_final",
        "boston_gov_es_file_document_files_2017_11_2017_homelesscensusresultspressrelease_171128",
        "boston_gov_sites_default_files_file_2023_09_pet_20rent_20policy_20_281_29_0",
        # Template with placeholder text, not real content
        "boston_gov_copy_20of_20official_20notice_20of_20accommodations_20_20template_20_20lca",
        # Multilingual housing search guide (mostly non-English translations)
        "boston_gov_departments_housing_office_housing_stability_departments_housing_office_housing_stability_housing_search_guid",
        # Contact pages with no substantive legal info
        "boston_gov_contact_inspectional_services",
        "boston_gov_departments_inspectional_services",
        # Homeowner-focused
        "boston_gov_departments_housing_boston_home_center_quick_guide_home_repair_programs",
        "boston_gov_departments_housing_how_city_creates_affordable_housing",
        "boston_gov_departments_housing_neighborhood_homes_initiative",
        "boston_gov_departments_housing_welcome_home_boston",
        # Policy/research pages (links only, no substance)
        "boston_gov_departments_housing_policies",
        "boston_gov_departments_housing_policy_development_and_research",
        # Homelessness (not tenant law)
        "boston_gov_departments_housing_ending_homelessness",
        "boston_gov_departments_housing_services_those_experiencing_homeless",
        # Zoning/development policy
        "boston_gov_departments_housing_inclusionary_zoning",
        # Navigation/index page with no substantive legal content
        "boston_gov_contact_housing",
        # Emergency shelter (homelessness, not tenant law)
        "mass_gov_www_mass_gov_how_to_review_eligibility_apply_for_emergency_assistance_ea_family_shelter",
        "mass_gov_www_mass_gov_info_details_what_is_emergency_assistance_ea_family_shelter",
    }
    before = len(chunks)
    chunks = [c for c in chunks if c["doc_id"] not in off_topic_docs]
    print(f"  Remove off-topic docs: {before} -> {len(chunks)} (-{before - len(chunks)})")
    return chunks


# ---------------------------------------------------------------------------
# Chunk-level filters
# ---------------------------------------------------------------------------

def remove_non_english_chunks(chunks: list[dict]) -> list[dict]:
    """Remove individual chunks with predominantly non-English text."""
    before = len(chunks)
    result = []
    for c in chunks:
        content = c["content"]
        non_ascii = sum(1 for ch in content if ord(ch) > 127)
        # If >15% non-ASCII characters, likely non-English
        if len(content) > 50 and non_ascii / len(content) > 0.15:
            continue
        result.append(c)
    print(f"  Remove non-English chunks: {before} -> {len(result)} (-{before - len(result)})")
    return result


def remove_homebuyer_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks focused on homebuying (not renting) from mixed-topic docs."""
    before = len(chunks)
    result = []
    for c in chunks:
        content = c["content"]
        # Only filter chunks that are clearly homebuyer-only (not mixed tenant/buyer pages)
        homebuyer_signals = len(re.findall(
            r"homebuyer|home buyer|buying a home|homeownership|homebuying|HomeBuyer",
            content, re.IGNORECASE
        ))
        tenant_signals = len(re.findall(
            r"tenant|renter|landlord|evict|lease|rent(?:al|er|ing)|deposit",
            content, re.IGNORECASE
        ))
        if homebuyer_signals >= 2 and tenant_signals == 0:
            continue
        result.append(c)
    print(f"  Remove homebuyer-only chunks: {before} -> {len(result)} (-{before - len(result)})")
    return result


def remove_endnote_only_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks that are just endnote numbers, URLs, or citation lists."""
    before = len(chunks)
    result = []
    for c in chunks:
        content = c["content"].strip()
        words = content.split()
        if len(words) < 5:
            # Very short — check if it's just a number + URL or citation ref
            stripped = re.sub(r"[\d\s.,;:()\-–—/]", "", content)
            stripped = re.sub(r"https?://\S+", "", stripped)
            if len(stripped) < 10:
                continue
        result.append(c)
    print(f"  Remove endnote-only chunks: {before} -> {len(result)} (-{before - len(result)})")
    return result


def remove_nav_and_link_lists(chunks: list[dict]) -> list[dict]:
    """Remove chunks that are just navigation menus, link lists, or TOCs."""
    before = len(chunks)
    result = []
    for c in chunks:
        content = c["content"].strip()
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if not lines:
            continue

        # Pattern A: Nav-link pattern — many short lines, no sentences ending in periods
        non_heading = [l for l in lines if not l.startswith("#")]
        if (len(non_heading) >= 3
                and all(len(l) < 60 for l in non_heading)
                and len(content) < 600):
            no_period = sum(1 for l in non_heading if not l.rstrip().endswith("."))
            if no_period / len(non_heading) > 0.7:
                continue

        # Pattern B: Larger link-list / sitemap chunks
        # Exempt masslegalhelp docs (2-column PDF extraction creates short lines)
        if len(non_heading) >= 8 and not c["doc_id"].startswith("masslegalhelp_"):
            link_like = [l for l in non_heading if len(l) < 80 and l and l[-1] not in ".!?:;"]
            link_ratio = len(link_like) / len(non_heading)
            avg_line_len = sum(len(l) for l in non_heading) / len(non_heading)
            if link_ratio > 0.65 and avg_line_len < 45:
                continue

        # HTML time tag artifacts (news/events sections)
        if "<time datetime=" in content:
            continue

        # Newsletter signup / "Stay Connected" boilerplate
        if "sign up" in content.lower() and "newsletter" in content.lower() and len(content) < 300:
            continue

        # PDF table of contents (dotted lines with page numbers)
        dots = len(re.findall(r"\.{3,}", content))
        if dots >= 1 and len(content) < 600:
            continue

        result.append(c)
    print(f"  Remove nav/link/TOC chunks: {before} -> {len(result)} (-{before - len(result)})")
    return result


def remove_header_only_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks that are almost entirely headings with negligible body text."""
    before = len(chunks)
    result = []
    for c in chunks:
        content = c["content"].strip()
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if not lines:
            continue
        heading_lines = [l for l in lines if l.startswith("#")]
        body_lines = [l for l in lines if not l.startswith("#")]
        body_text = " ".join(body_lines).strip()
        # Remove if >75% headings AND body text < 50 chars
        if len(lines) >= 2 and len(heading_lines) / len(lines) > 0.75 and len(body_text) < 50:
            continue
        result.append(c)
    print(f"  Remove header-only chunks: {before} -> {len(result)} (-{before - len(result)})")
    return result


def merge_heading_chunks(chunks: list[dict]) -> list[dict]:
    """Merge heading-only chunks into the next substantive chunk in the same document."""
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    before = len(chunks)
    result = []
    for doc_id, doc_chunks in by_doc.items():
        doc_chunks.sort(key=lambda c: c["chunk_index"])
        merged = []
        pending_text = ""

        for chunk in doc_chunks:
            content = chunk["content"]
            is_heading = len(content.strip()) < 100 and (
                content.strip().startswith("#")
                or re.match(r"^[A-Z\s\d.:§()]+$", content.strip())
            )

            if is_heading:
                pending_text += ("\n\n" if pending_text else "") + content.strip()
            else:
                if pending_text:
                    chunk = dict(chunk)
                    chunk["content"] = pending_text + "\n\n" + content
                    pending_text = ""
                merged.append(chunk)

        if pending_text and not merged:
            last = doc_chunks[-1].copy()
            last["content"] = pending_text
            merged.append(last)
        elif pending_text and merged:
            merged[-1] = dict(merged[-1])
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + pending_text

        result.extend(merged)

    print(f"  Merge heading chunks: {before} -> {len(result)} (-{before - len(result)})")
    return result


def filter_short_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks shorter than MIN_CHUNK_LENGTH characters."""
    before = len(chunks)
    chunks = [c for c in chunks if len(c["content"].strip()) >= MIN_CHUNK_LENGTH]
    print(f"  Filter short chunks (<{MIN_CHUNK_LENGTH} chars): {before} -> {len(chunks)} (-{before - len(chunks)})")
    return chunks


def is_boilerplate(text: str) -> bool:
    """Detect navigation menus and boilerplate content."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return True

    if all(len(l) < 40 for l in lines) and len(text.strip()) < 150:
        return True

    boilerplate_phrases = [
        "upcoming events", "latest updates", "toggle", "page sections",
        "skip to main content", "back to top", "cookie", "subscribe to",
    ]
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in boilerplate_phrases) and len(text.strip()) < 200:
        return True

    return False


def remove_boilerplate(chunks: list[dict]) -> list[dict]:
    """Remove boilerplate/navigation chunks."""
    before = len(chunks)
    chunks = [c for c in chunks if not is_boilerplate(c["content"])]
    print(f"  Remove boilerplate: {before} -> {len(chunks)} (-{before - len(chunks)})")
    return chunks


def deduplicate_by_content(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks with identical or near-identical content."""
    seen_hashes = set()
    result = []
    before = len(chunks)
    for c in chunks:
        normalized = re.sub(r"\s+", " ", c["content"].strip().lower())
        content_hash = hashlib.md5(normalized.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            result.append(c)
    print(f"  Deduplicate by content: {before} -> {len(result)} (-{before - len(result)})")
    return result


def reindex_chunks(chunks: list[dict]) -> list[dict]:
    """Reassign chunk_index and chunk_id after filtering."""
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    result = []
    for doc_id, doc_chunks in by_doc.items():
        doc_chunks.sort(key=lambda c: c["chunk_index"])
        for i, chunk in enumerate(doc_chunks):
            chunk = dict(chunk)
            chunk["chunk_index"] = i
            chunk["total_chunks"] = len(doc_chunks)
            chunk["chunk_id"] = f"{doc_id}_chunk_{i:03d}"
            result.append(chunk)

    return result


def run():
    """Load, clean, save, and print stats."""
    print("=" * 60)
    print("Corpus Cleaner")
    print("=" * 60)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    # Document-level filters
    chunks = remove_faq_full_page(chunks)
    chunks = remove_non_english_docs(chunks)
    chunks = remove_oversized_reports(chunks)
    chunks = remove_off_topic_docs(chunks)

    # Chunk-level merging
    chunks = merge_heading_chunks(chunks)

    # Chunk-level filters
    chunks = remove_non_english_chunks(chunks)
    chunks = remove_homebuyer_chunks(chunks)
    chunks = remove_endnote_only_chunks(chunks)
    chunks = remove_nav_and_link_lists(chunks)
    chunks = remove_header_only_chunks(chunks)
    chunks = filter_short_chunks(chunks)
    chunks = remove_boilerplate(chunks)
    chunks = deduplicate_by_content(chunks)

    # Reindex
    chunks = reindex_chunks(chunks)

    save_chunks(chunks)

    # Verify
    faq_full = [c for c in chunks if c["doc_id"] == "bostonhousing_faq_full_page"]
    short = [c for c in chunks if len(c["content"].strip()) < MIN_CHUNK_LENGTH]
    print(f"\n{'=' * 60}")
    print(f"Final chunk count: {len(chunks)}")
    print(f"FAQ full page chunks remaining: {len(faq_full)}")
    print(f"Short chunks remaining (<{MIN_CHUNK_LENGTH} chars): {len(short)}")
    print(f"Saved to: {CHUNKS_PATH}")
    print(f"{'=' * 60}")

    return chunks


if __name__ == "__main__":
    run()
