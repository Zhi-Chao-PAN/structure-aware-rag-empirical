"""
Data Parsing Pipeline

Compares multiple PDF parsing methods for financial document extraction:
- PyPDF2: Baseline unstructured text extraction
- pdfplumber: Open-source with table detection
- LlamaParse: State-of-the-art structure-aware parsing (API-based)

Author: Zhichao Pan
Version: 1.1.0
"""
from __future__ import annotations

import os
import sys
import hashlib
import logging
from typing import Optional, List
from pathlib import Path

import requests
import pypdf
import pdfplumber
from llama_parse import LlamaParse
from dotenv import load_dotenv
import nest_asyncio

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import data_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
nest_asyncio.apply()

def get_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ensure_dirs() -> None:
    """Create necessary directory structure."""
    os.makedirs(data_config.RAW_DIR, exist_ok=True)
    os.makedirs(os.path.join(data_config.PARSED_DIR, "pypdf"), exist_ok=True)
    os.makedirs(os.path.join(data_config.PARSED_DIR, "pdfplumber"), exist_ok=True)
    os.makedirs(os.path.join(data_config.PARSED_DIR, "llamaparse"), exist_ok=True)

def download_pdf() -> Optional[str]:
    """Download PDF with hash verification."""
    path = os.path.join(data_config.RAW_DIR, data_config.PDF_FILENAME)
    
    if os.path.exists(path):
        logger.info(f"✅ PDF exists at {path}")
        # Verify hash
        current_hash = get_file_hash(path)
        if hasattr(data_config, 'PDF_EXPECTED_HASH') and data_config.PDF_EXPECTED_HASH:
             if current_hash != data_config.PDF_EXPECTED_HASH:
                 logger.warning(f"⚠️ Hash mismatch! Expected {data_config.PDF_EXPECTED_HASH}, got {current_hash}")
                 logger.warning("   The file may be corrupted or replaced. Proceeding with caution.")
             else:
                 logger.info("✅ Hash verified.")
        else:
             logger.info(f"ℹ️ File Hash: {current_hash} (Add to config/data_config.py to enforce)")
        return path
    
    logger.info(f"⬇️ Downloading {data_config.PDF_FILENAME}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(data_config.PDF_URL, headers=headers, timeout=60)
        response.raise_for_status()
        
        with open(path, 'wb') as f:
            f.write(response.content)
            
        logger.info("✅ Download complete.")
        logger.info(f"ℹ️ File Hash: {get_file_hash(path)}")
        return path
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return None

def run_pypdf(pdf_path: str) -> None:
    """Run PyPDF baseline extraction."""
    try:
        output_path = os.path.join(data_config.PARSED_DIR, "pypdf", "parsed.md")
        if os.path.exists(output_path):
            logger.info("⏩ PyPDF output exists. Skipping.")
            return

        logger.info("🏃 Running PyPDF (Baseline)...")
        text = "# PyPDF Baseline Output\n\n"
        reader = pypdf.PdfReader(pdf_path)
        
        for i in data_config.TARGET_PAGES:
            text += f"## Page {i+1}\n\n"
            text += reader.pages[i].extract_text() + "\n\n"
            text += "---\n\n"
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"✅ Saved to {output_path}")
    except Exception as e:
        logger.error(f"❌ PyPDF Failed: {e}")

def run_pdfplumber(pdf_path: str) -> None:
    """Run pdfplumber table extraction."""
    try:
        output_path = os.path.join(data_config.PARSED_DIR, "pdfplumber", "parsed.md")
        if os.path.exists(output_path):
            logger.info("⏩ pdfplumber output exists. Skipping.")
            return

        logger.info("🏃 Running pdfplumber (Open Source)...")
        text = "# pdfplumber Output\n\n"
        
        with pdfplumber.open(pdf_path) as pdf:
            for i in data_config.TARGET_PAGES:
                text += f"## Page {i+1}\n\n"
                page = pdf.pages[i]
                
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        cleaned_table = [
                            [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                            for row in table
                        ]
                        if len(cleaned_table) > 0:
                            header = "| " + " | ".join(cleaned_table[0]) + " |"
                            separator = "| " + " | ".join(["---"] * len(cleaned_table[0])) + " |"
                            body = "\n".join(["| " + " | ".join(row) + " |" for row in cleaned_table[1:]])
                            text += f"\n{header}\n{separator}\n{body}\n\n"
                
                text += "\n*(Raw text extraction skipped for clarity)*\n"
                text += "---\n\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"✅ Saved to {output_path}")
    except Exception as e:
        logger.error(f"❌ pdfplumber Failed: {e}")

def run_llamaparse(pdf_path: str) -> None:
    """Run LlamaParse SOTA extraction."""
    try:
        output_path = os.path.join(data_config.PARSED_DIR, "llamaparse", "parsed.md")
        if os.path.exists(output_path):
            logger.info("⏩ LlamaParse output exists. Skipping.")
            return

        logger.info("🏃 Running LlamaParse (SOTA)...")
        
        api_key: Optional[str] = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key or "PLACEHOLDER" in api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY missing or invalid in .env")

        target_pages_str = ",".join(map(str, data_config.TARGET_PAGES))

        # Important: Privacy Warning (Dimension 4, Point 22)
        logger.warning("⚠️  UPLOADING DATA TO LLAMA_CLOUD. Ensure this is safe for your use-case.")
        
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
        logger.info(f"✅ Saved to {output_path}")
        
    except ValueError as ve:
        logger.error(f"❌ Configuration Error: {ve}")
    except Exception as e:
        logger.error(f"❌ LlamaParse Failed: {e}")

if __name__ == "__main__":
    ensure_dirs()
    pdf_path = download_pdf()
    
    if pdf_path:
        # Process Isolation: Sequentially run with independent error handling
        run_pypdf(pdf_path)
        run_pdfplumber(pdf_path)
        run_llamaparse(pdf_path)
        
        print("\n🎉 Pipeline finished. Check 'data/parsed' for outputs.")
    else:
        logger.error("❌ Could not proceed without PDF.")
