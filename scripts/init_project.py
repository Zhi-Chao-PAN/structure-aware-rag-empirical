"""
Structure-Aware RAG Study - Project Initialization
Creates directory structure and base configuration files.

Author: Zhichao Pan
Version: 1.0.0
"""
import os

# 1. Create directory structure
directories = [
    "src/parsing",
    "src/experiments",
    "src/evaluation",
    "data/raw_pdfs",
    "data/parsed",
    "data/benchmark",
    "data/chroma_db",
    "experiments",
    "report"
]

for d in directories:
    os.makedirs(d, exist_ok=True)
    print(f"Verified directory: {d}")

# 2. Create .gitignore
gitignore_content = """# Python artifacts
__pycache__/
*.pyc
.ipynb_checkpoints/

# Virtual environments
venv/
.venv/
env/

# Environment variables (NEVER upload API keys!)
.env

# Data files (only upload benchmark and sample PDFs, not large built indices)
data/chroma_db/
data/parsed/

# IDE configurations
.vscode/
.idea/
"""

with open(".gitignore", "w", encoding="utf-8") as f:
    f.write(gitignore_content)
    print("Created .gitignore")
