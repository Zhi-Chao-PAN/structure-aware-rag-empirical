"""
Data Parsing Pipeline

Compares multiple PDF parsing methods for financial document extraction:
- PyPDF2: Baseline unstructured text extraction
- pdfplumber: Open-source with table detection
- LlamaParse: State-of-the-art structure-aware parsing (API-based)

This module downloads a sample financial document (NVIDIA 10-K) and processes
it through each parser, saving outputs in Markdown format for comparison.

Author: Zhichao Pan
Version: 1.0.0
"""
from __future__ import annotations

import os
from typing import Optional

import requests
import pypdf
import pdfplumber
from llama_parse import LlamaParse
from dotenv import load_dotenv
import nest_asyncio

# Load environment variables (.env)
load_dotenv()
nest_asyncio.apply()

# ================= Configuration =================
PDF_URL: str = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf"
PDF_FILENAME: str = "nvidia_2024_10k.pdf"
RAW_DIR: str = "data/raw_pdfs"
PARSED_DIR: str = "data/parsed"

# Important optimization: Only parse pages containing key financial tables to save API quota
# Page 34 (index 33): Consolidated Statements of Income
# Page 35 (index 34): Consolidated Statements of Comprehensive Income
# Page 36 (index 35): Consolidated Balance Sheets
TARGET_PAGES: list[int] = [33, 34, 35]


# ================= Utility Functions =================

def ensure_dirs() -> None:
    """Create necessary directory structure for parsed outputs."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.join(PARSED_DIR, "pypdf"), exist_ok=True)
    os.makedirs(os.path.join(PARSED_DIR, "pdfplumber"), exist_ok=True)
    os.makedirs(os.path.join(PARSED_DIR, "llamaparse"), exist_ok=True)


def download_pdf() -> str:
    """
    Download the source PDF if it doesn't already exist locally.
    
    Returns:
        str: Path to the downloaded PDF file.
    """
    path = os.path.join(RAW_DIR, PDF_FILENAME)
    if os.path.exists(path):
        print(f"✅ PDF already exists at {path}")
        return path
    
    print(f"⬇️ Downloading {PDF_FILENAME}...")
    headers = {'User-Agent': 'Mozilla/5.0'}  # Mimic browser to avoid 403
    response = requests.get(PDF_URL, headers=headers, timeout=60)
    response.raise_for_status()
    
    with open(path, 'wb') as f:
        f.write(response.content)
    print("✅ Download complete.")
    return path


# ================= Parsing Logic =================

def run_pypdf(pdf_path: str) -> None:
    """
    Run PyPDF baseline extraction.
    
    Extracts raw text from specified pages without any structure preservation.
    This serves as the control condition in our experiment.
    
    Args:
        pdf_path: Path to the source PDF file.
    """
    output_path = os.path.join(PARSED_DIR, "pypdf", "parsed.md")
    if os.path.exists(output_path):
        print("⏩ PyPDF output exists. Skipping.")
        return

    print("🏃 Running PyPDF (Baseline)...")
    text = "# PyPDF Baseline Output\n\n"
    reader = pypdf.PdfReader(pdf_path)
    
    for i in TARGET_PAGES:
        text += f"## Page {i+1}\n\n"
        text += reader.pages[i].extract_text() + "\n\n"
        text += "---\n\n"
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved to {output_path}")


def run_pdfplumber(pdf_path: str) -> None:
    """
    Run pdfplumber table extraction.
    
    Attempts to detect and extract tables, converting them to Markdown format.
    Uses open-source computer vision methods for table detection.
    
    Args:
        pdf_path: Path to the source PDF file.
    """
    output_path = os.path.join(PARSED_DIR, "pdfplumber", "parsed.md")
    if os.path.exists(output_path):
        print("⏩ pdfplumber output exists. Skipping.")
        return

    print("🏃 Running pdfplumber (Open Source)...")
    text = "# pdfplumber Output\n\n"
    
    with pdfplumber.open(pdf_path) as pdf:
        for i in TARGET_PAGES:
            text += f"## Page {i+1}\n\n"
            page = pdf.pages[i]
            
            # Extract tables and convert to Markdown
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    # Clean data: remove None, convert to string
                    cleaned_table = [
                        [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                        for row in table
                    ]
                    # Build Markdown table header and separator
                    if len(cleaned_table) > 0:
                        header = "| " + " | ".join(cleaned_table[0]) + " |"
                        separator = "| " + " | ".join(["---"] * len(cleaned_table[0])) + " |"
                        body = "\n".join(["| " + " | ".join(row) + " |" for row in cleaned_table[1:]])
                        text += f"\n{header}\n{separator}\n{body}\n\n"
            
            # Extract remaining text (simplified for demo)
            text += "\n*(Raw text extraction skipped for clarity in this demo)*\n"
            text += "---\n\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved to {output_path}")


def run_llamaparse(pdf_path: str) -> None:
    """
    Run LlamaParse SOTA extraction.
    
    Uses LlamaIndex's cloud-based parsing service to extract documents
    with full structure preservation in Markdown format. Requires API key.
    
    Args:
        pdf_path: Path to the source PDF file.
    """
    output_path = os.path.join(PARSED_DIR, "llamaparse", "parsed.md")

    print("🏃 Running LlamaParse (SOTA)...")
    
    api_key: Optional[str] = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key or "PLACEHOLDER" in api_key or "your_api_key" in api_key.lower():
        print("❌ Error: LLAMA_CLOUD_API_KEY not found or is placeholder in .env")
        print("   Please add your API key to .env file.")
        return

    try:
        # Convert list [33, 34, 35] to string "33,34,35"
        target_pages_str = ",".join(map(str, TARGET_PAGES))

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            target_pages=target_pages_str,
            verbose=True,
            language="en"
        )
        
        documents = parser.load_data(pdf_path)
        text = "# LlamaParse SOTA Output\n\n"
        for doc in documents:
            text += doc.text + "\n\n---\n\n"
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Saved to {output_path}")
    except Exception as e:
        print(f"❌ LlamaParse failed: {e}")


# ================= Main Entry =================

if __name__ == "__main__":
    ensure_dirs()
    pdf_path = download_pdf()
    
    run_pypdf(pdf_path)
    run_pdfplumber(pdf_path)
    run_llamaparse(pdf_path)
    
    print("\n🎉 Pipeline check finished. Check 'data/parsed' for available outputs.")
