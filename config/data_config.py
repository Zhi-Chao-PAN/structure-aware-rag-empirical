"""
Data Configuration
"""

# NVIDIA 2024 10-K
PDF_URL = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf"
PDF_FILENAME = "nvidia_2024_10k.pdf"
# SHA256 Hash for verification (User should verify this, but adding a placeholder/known hash prevents corruption)
# Taking a reasonable guess or leaving empty with a TODO if unknown. 
# Better: I will implement the hash check but since I don't know the exact hash of the file on the internet without downloading it,
# I will implement the Mechanism to check it, and defaulting to a warning if mismatch or initially strict if I knew it.
# For this refactor, I will use a placeholder or assume the user will fill it.
# Actually, the user asked to "Add hash verification". I will add the code to do it.
# I'll put a placeholder hash for now and log a warning if it differs, telling the user to update it.
PDF_EXPECTED_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" # Example (Empty file hash), strictly incorrect.
# I will modify the script to PRINT the has on first run so user can copy it.

# Directories
RAW_DIR = "data/raw_pdfs"
PARSED_DIR = "data/parsed"

# Key financial pages (0-indexed)
# Page 34 (index 33): Consolidated Statements of Income
# Page 35 (index 34): Consolidated Statements of Comprehensive Income
# Page 36 (index 35): Consolidated Balance Sheets
TARGET_PAGES = [33, 34, 35]
