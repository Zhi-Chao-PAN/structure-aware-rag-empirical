import pandas as pd
import re
import os

INPUT_FILE = "experiments/comparison_results.csv"
OUTPUT_FILE = "experiments/scored_results.csv"

def auto_score(row):
    """
    A simple auto-grader to save manual effort.
    Logic: If numbers from Ground Truth appear in Model Answer, give credit.
    """
    ans = str(row['model_answer'])
    # Remove <think> traces, only keep final answer
    ans = re.sub(r'<think>.*?</think>', '', ans, flags=re.DOTALL).lower()
    gt = str(row['ground_truth']).lower()
    
    # Extract all numbers from GT (e.g., "60,922" -> ["60", "922"] or "60922")
    # Better regex to capture numbers with commas or decimals
    gt_nums = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', gt)
    
    if not gt_nums:
        # If GT is text-only (e.g. "Yes")
        return 1.0 if gt in ans else 0.0
    
    # Check number hits
    hits = 0
    total_nums = len(gt_nums)
    for num in gt_nums:
        clean_num = num.replace(',', '') # Remove comma for matching
        # Check if the clean number exists in the answer (ignoring commas in answer too)
        if clean_num in ans.replace(',', ''):
            hits += 1
            
    if hits == total_nums:
        return 1.0 # Perfect match
    elif hits > 0:
        return 0.5 # Partial match
    else:
        return 0.0 # No match

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        df = pd.read_csv(INPUT_FILE)
        print("🤖 Running Auto-Grader...")
        df['score'] = df.apply(auto_score, axis=1)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Pre-filled scores saved to: {OUTPUT_FILE}")
        print("👉 Action: Open this file and verify the scores manually!")
    else:
        print(f"❌ Data file not found: {INPUT_FILE}")
