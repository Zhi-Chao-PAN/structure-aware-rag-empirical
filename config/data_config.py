"""
Data Configuration
"""

# NVIDIA 2024 10-K
PDF_URL = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf"
PDF_FILENAME = "nvidia_2024_10k.pdf"
# SHA256 Hash for verification
PDF_EXPECTED_HASH = "jOZMvGgkuKhrAigJI36bUvIWBxLlqHefb7JRms16YqU"
# I will modify the script to PRINT the has on first run so user can copy it.

# Directories
RAW_DIR = "data/raw_pdfs"
PARSED_DIR = "data/parsed"

# Key financial pages (0-indexed)
# Page 34 (index 33): Consolidated Statements of Income
# Page 35 (index 34): Consolidated Statements of Comprehensive Income
# Page 36 (index 35): Consolidated Balance Sheets
TARGET_PAGES = [33, 34, 35]
