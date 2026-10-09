import pandas as pd
import matplotlib
# Use Agg backend for headless environments (Dimension 4, Point 28)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import Optional

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def generate_plots(csv_path: str = "experiments/scored_results.csv", output_dir: str = "report") -> None: 
    """
    Generate evaluation plots from scored results.
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ File {csv_path} not found. Please score the results first!")
        return

    # robust encoding handling (Dimension 4, Point 24)
    df = None
    for encoding in ['utf-8-sig', 'utf-8', 'latin1']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
            
    if df is None:
        print("❌ Failed to read CSV with utf-8 or latin1 encoding.")
        return
    
    # Ensure 'score' column is numeric
    if 'score' not in df.columns:
        print("⚠️ 'score' column missing in the CSV.")
        return
        
    # Output Dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Accuracy Comparison
    plt.figure(figsize=(8, 6))
    
    # Check if pipeline column exists
    if 'pipeline' in df.columns:
        acc_df = df.groupby('pipeline')['score'].mean().reset_index()
        
        # Dynamic plotting
        sns.barplot(data=acc_df, x='pipeline', y='score', hue='pipeline', legend=False)
        plt.title('Accuracy by Pipeline (Human/Auto Eval)', fontsize=14, fontweight='bold')
        plt.ylabel('Average Score (0-1)', fontsize=12)
        plt.ylim(0, 1.1)
        
        for index, row in acc_df.iterrows():
            plt.text(index, row.score + 0.02, f"{row.score:.2%}", ha='center', fontweight='bold')
            
        out_path = os.path.join(output_dir, 'accuracy_comparison.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✅ Generated: {out_path}")
        plt.close()

        # 2. Latency Distribution
        if 'latency_s' in df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x='pipeline', y='latency_s', hue='pipeline', legend=False)
            plt.title('Inference Latency Distribution', fontsize=14)
            plt.ylabel('Time per Question (seconds)', fontsize=12)
            
            out_path_lat = os.path.join(output_dir, 'latency_distribution.png')
            plt.savefig(out_path_lat, dpi=300, bbox_inches='tight')
            print(f"✅ Generated: {out_path_lat}")
            plt.close()
    else:
        print("⚠️ 'pipeline' column missing. Cannot compare groups.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visualization")
    parser.add_argument("--input", default="experiments/scored_results.csv", help="Input CSV")
    parser.add_argument("--output", default="report", help="Output directory")
    args = parser.parse_args()
    
    generate_plots(args.input, args.output)
