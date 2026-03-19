"""
Replace Cornell Law regulation sources with official mass.gov PDF versions.

Replaces:
  1. 940 CMR 3.17 (Cornell → mass.gov PDF, pages 1-3 + 18-22 for definitions + s.3.17)
  2. 105 CMR 410 (Cornell → mass.gov PDF, all 34 pages — full sanitary code)
"""

import os
import re
import sys
from pathlib import Path

import pdfplumber

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scraping.utils import (
    extract_legal_citations,
    content_hash,
    save_document,
    PROCESSED_DIR,
)

# PDF paths (already downloaded)
PDF_940_CMR = Path(
    "/Users/eps/.claude/projects/-Users-eps-Desktop-Work-NEU-Spring-2026-CS6180-Final-Project/"
    "5807c4a0-8bd3-4bfc-9056-9216ce965866/tool-results/webfetch-1773717651082-ht8853.pdf"
)
PDF_105_CMR = Path(
    "/Users/eps/.claude/projects/-Users-eps-Desktop-Work-NEU-Spring-2026-CS6180-Final-Project/"
    "5807c4a0-8bd3-4bfc-9056-9216ce965866/tool-results/webfetch-1773717741120-jv8g82.pdf"
)

# Old Cornell-sourced files to delete
OLD_940 = PROCESSED_DIR / "mass_gov_www_law_cornell_edu_regulations_massachusetts_940_3_17.json"
OLD_105 = PROCESSED_DIR / "mass_gov_www_law_cornell_edu_regulations_massachusetts_department_105_title_105_410_000.json"


def extract_pdf_text(pdf_path: Path, pages: list[int] | None = None) -> str:
    """Extract text from a PDF, optionally from specific pages (0-indexed)."""
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        target_pages = pages if pages is not None else range(len(pdf.pages))
        for i in target_pages:
            page = pdf.pages[i]
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def clean_regulation_text(text: str) -> str:
    """Clean extracted PDF text: fix spacing, remove repeated headers/footers."""
    # Remove page-level headers that repeat on every page
    text = re.sub(r"940 CMR: OFFICE OF THE ATTORNEY GENERAL\n?", "", text)
    text = re.sub(r"105 CMR: DEPARTMENT OF PUBLIC HEALTH\n?", "", text)

    # Remove Mass Register footer lines
    text = re.sub(r"\(Mass\. Register #\d+.*?\)\n?", "", text)

    # Fix hyphenation at line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Clean up lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def format_as_markdown(text: str, title: str) -> str:
    """Add markdown structure to regulation text."""
    # Add title as h1
    result = f"# {title}\n\n{text}"

    # Convert section numbers to markdown headers
    # Match patterns like "3.01: Definitions" or "410.010: Definitions"
    result = re.sub(
        r"^(\d+\.\d+(?:\.\d+)?):?\s+([A-Z].*?)$",
        r"## \1: \2",
        result,
        flags=re.MULTILINE,
    )

    # Convert "Section: continued" to subheader
    result = re.sub(
        r"^(\d+\.\d+(?:\.\d+)?):?\s+continued$",
        r"## \1 (continued)",
        result,
        flags=re.MULTILINE,
    )

    return result


def replace_940_cmr():
    """Replace 940 CMR 3.17 with official mass.gov PDF source."""
    print("\n=== Replacing 940 CMR 3.17 ===")

    # Extract pages: 0-2 (TOC + definitions) and 17-21 (section 3.17)
    # Page numbers are 0-indexed in pdfplumber
    pages = list(range(0, 3)) + list(range(17, 22))
    raw_text = extract_pdf_text(PDF_940_CMR, pages)
    cleaned = clean_regulation_text(raw_text)
    content = format_as_markdown(
        cleaned,
        "940 CMR 3.17 — Landlord-Tenant Consumer Protection Regulations",
    )

    print(f"  Extracted {len(content)} chars from pages {[p+1 for p in pages]}")

    # Delete old file
    if OLD_940.exists():
        OLD_940.unlink()
        print(f"  Deleted: {OLD_940.name}")
    else:
        print(f"  [WARN] Old file not found: {OLD_940.name}")

    # Save new document
    doc_id = "mass_gov_940_cmr_3_17_landlord_tenant"
    save_document(
        doc_id=doc_id,
        source_url="https://www.mass.gov/doc/940-cmr-3-consumer-protection-general-regulations/download",
        source_name="mass_gov",
        title="940 CMR 3.17 — Landlord-Tenant Consumer Protection Regulations",
        content=content,
        content_type="regulation",
        section_headers=[
            "940 CMR 3.00: General Regulations",
            "3.01: Definitions",
            "3.17: Landlord-Tenant",
            "3.17(1): Conditions and Maintenance of a Dwelling Unit",
            "3.17(2): Notices and Demands",
            "3.17(3): Rental Agreements",
            "3.17(4): Security Deposits and Rent in Advance",
            "3.17(5): Evictions and Termination of Tenancy",
            "3.17(6): Miscellaneous",
        ],
    )
    print("  Done: 940 CMR 3.17 replaced with mass.gov source")


def replace_105_cmr():
    """Replace 105 CMR 410 with official mass.gov PDF source (full regulation)."""
    print("\n=== Replacing 105 CMR 410 ===")

    # Extract ALL pages (full regulation)
    raw_text = extract_pdf_text(PDF_105_CMR)
    cleaned = clean_regulation_text(raw_text)
    content = format_as_markdown(
        cleaned,
        "105 CMR 410 — Minimum Standards of Fitness for Human Habitation (State Sanitary Code Chapter II)",
    )

    print(f"  Extracted {len(content)} chars from all 34 pages")

    # Delete old file
    if OLD_105.exists():
        OLD_105.unlink()
        print(f"  Deleted: {OLD_105.name}")
    else:
        print(f"  [WARN] Old file not found: {OLD_105.name}")

    # Save new document
    doc_id = "mass_gov_105_cmr_410_sanitary_code"
    save_document(
        doc_id=doc_id,
        source_url="https://www.mass.gov/doc/105-cmr-410-minimum-standards-of-fitness-for-human-habitation-state-sanitary-code-chapter-ii/download",
        source_name="mass_gov",
        title="105 CMR 410 — Minimum Standards of Fitness for Human Habitation (State Sanitary Code Chapter II)",
        content=content,
        content_type="regulation",
        section_headers=[
            "105 CMR 410.000: State Sanitary Code Chapter II",
            "410.001: Purpose",
            "410.002: Scope",
            "410.003: General Provisions",
            "410.010: Definitions",
            "410.100: Kitchen Facilities",
            "410.150: Hot Water",
            "410.160: Heating Systems",
            "410.180: Temperature Requirements",
            "410.200: Provision and Metering of Electricity or Gas",
            "410.250: Asbestos-containing Material",
            "410.260: Means of Egress",
            "410.300: Electricity Supply and Illumination",
            "410.330: Smoke Detectors and Carbon Monoxide Alarms",
            "410.400: Owner/Manager Contact Information",
            "410.420: Habitability Requirements",
            "410.470: Lead-based Paint Hazards",
            "410.500: Owner's Responsibility to Maintain Building",
            "410.550: Elimination of Pests",
            "410.630: Conditions Deemed to Endanger Health or Safety",
        ],
    )
    print("  Done: 105 CMR 410 replaced with mass.gov source (full regulation)")


if __name__ == "__main__":
    print("Replacing Cornell Law regulation sources with official mass.gov PDFs...")
    print(f"Processed dir: {PROCESSED_DIR}")

    # Count docs before
    before_count = len(list(PROCESSED_DIR.glob("*.json")))
    print(f"Documents before: {before_count}")

    replace_940_cmr()
    replace_105_cmr()

    # Count docs after
    after_count = len(list(PROCESSED_DIR.glob("*.json")))
    print(f"\nDocuments after: {after_count}")
    print(f"Net change: {after_count - before_count:+d}")
    print("\nDone! Next steps:")
    print("  1. Run chunker:  venv/bin/python3 -m src.processing.chunker")
    print("  2. Re-index:     venv/bin/python3 -c \"from src.rag.pipeline import index_chunks; index_chunks()\"")
    print("  3. Sanity check: venv/bin/python3 -c \"from src.rag.pipeline import sanity_check; sanity_check()\"")
