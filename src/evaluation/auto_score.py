import pandas as pd
import re
import os
import argparse
import logging
from typing import List, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = "experiments/comparison_results.csv"
OUTPUT_FILE = "experiments/scored_results.csv"

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, remove punctuation except valid number chars.
    """
    if not isinstance(text, str):
        return str(text).lower().strip()
    
    # Remove <think> blocks if present (DeepSeek style)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    return text.lower().strip()

def extract_numbers(text: str) -> List[str]:
    """
    Extract numbers from text, handling commas and decimals.
    Returns a list of normalized number strings (digits and optional dot).
    """
    # Pattern to match numbers: 
    # 1. digits with optional commas and decimals (e.g., 1,234.56)
    # 2. explicit notation like 60.9B is harder without a library, 
    #    so we blindly extract the numeric part "60.9" which is a minimal requirement.
    
    # Matches: 123 | 1,234 | 1.23 | 1,234.56
    matches = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', text)
    
    # Normalize: remove commas
    cleaned = [m.replace(',', '') for m in matches]
    return cleaned

def check_number_match(gt_nums: List[str], ans_text: str) -> float:
    """
    Check if all ground truth numbers appear in the answer.
    """
    if not gt_nums:
        return 0.0
        
    ans_nums = set(extract_numbers(ans_text))
    
    hits = 0
    for gt_num in gt_nums:
        # Direct match in extracted numbers
        if gt_num in ans_nums:
            hits += 1
            continue
            
        # Fallback: Substring match for cases like "60.9" inside "60.9B"
        # We search for the number ensuring it's not part of a larger number (e.g. 60 inside 160)
        # but 60.9 inside 60.9B is okay.
        
        # Simple heuristic: is the exact float value close? 
        # (Skipping advanced semantic parsing for now to keep dependencies low)
        pass 

    if hits == len(gt_nums):
        return 1.0
    elif hits > 0:
        return 0.5
    return 0.0

def check_text_match(gt: str, ans: str) -> float:
    """
    Check for text match using word boundaries to avoid 'No' matching 'Not'.
    """
    # Tokenize by non-alphanumeric
    gt_tokens = set(re.split(r'\W+', gt))
    ans_tokens = set(re.split(r'\W+', ans))
    
    # Filter empty
    gt_tokens = {t for t in gt_tokens if t}
    ans_tokens = {t for t in ans_tokens if t}
    
    if not gt_tokens:
        return 0.0
        
    # Check if critical GT keywords match
    common = gt_tokens.intersection(ans_tokens)
    
    # Logic: If GT is "Yes", Ans should likely contain "Yes".
    # If GT is "60 million", and we are here, numbers failed or weren't present.
    
    if len(common) == len(gt_tokens):
        return 1.0
    elif len(common) > 0:
        return 0.5
    return 0.0

def auto_score(row: pd.Series) -> float:
    """
    Auto-grader with improved logic/robustness.
    """
    ans = normalize_text(row['model_answer'])
    gt = normalize_text(row['ground_truth'])
    
    # 1. Extract numbers from GT
    gt_nums = extract_numbers(gt)
    
    if gt_nums:
        # Numeric comparison primary
        score = check_number_match(gt_nums, ans)
        if score == 1.0:
            return 1.0
        # If numeric check failed or partial, try text check as fallback/boost?
        # Actually financial data usually relies on numbers. 
        return score
    else:
        # Text-only comparison (e.g., "Yes", "High Risk")
        return check_text_match(gt, ans)

def main():
    parser = argparse.ArgumentParser(description="Auto-Scorer for RAG Results")
    parser.add_argument("--input", default=INPUT_FILE, help="Path to input CSV")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Path to output CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"❌ Input file not found: {args.input}")
        return

    logger.info(f"📂 Loading results from {args.input}...")
    df = pd.read_csv(args.input)
    
    logger.info("🤖 Running Auto-Grader...")
    # Ensure 'score' column exists
    df['score'] = df.apply(auto_score, axis=1)
    
    df.to_csv(args.output, index=False)
    logger.info(f"✅ Scored results saved to: {args.output}")
    logger.info("👉 Action: Open this file and verify the scores manually! Auto-graders are imperfect.")

if __name__ == "__main__":
    main()
