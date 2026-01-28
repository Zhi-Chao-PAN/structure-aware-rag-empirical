import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def generate_plots(csv_path="experiments/scored_results.csv"): 
    if not os.path.exists(csv_path):
        print(f"⚠️ File {csv_path} not found. Please score the results first!")
        return

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 decode failed, trying 'latin1'...")
        df = pd.read_csv(csv_path, encoding='latin1')
    
    # Ensure 'score' column is numeric
    if 'score' not in df.columns:
        print("⚠️ 'score' column missing in the CSV.")
        return
        
    # Output Dir
    os.makedirs('report', exist_ok=True)

    # 1. Accuracy Comparison
    plt.figure(figsize=(8, 6))
    acc_df = df.groupby('pipeline')['score'].mean().reset_index()
    
    sns.barplot(data=acc_df, x='pipeline', y='score', hue='pipeline', legend=False)
    plt.title('Accuracy by Pipeline (Human Eval)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Score (0-1)', fontsize=12)
    plt.ylim(0, 1.1)
    
    for index, row in acc_df.iterrows():
        plt.text(index, row.score + 0.02, f"{row.score:.2%}", ha='center', fontweight='bold')
        
    plt.savefig('report/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Generated: report/accuracy_comparison.png")

    # 2. Latency Distribution
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='pipeline', y='latency_s', hue='pipeline', legend=False)
    plt.title('Inference Latency Distribution', fontsize=14)
    plt.ylabel('Time per Question (seconds)', fontsize=12)
    plt.savefig('report/latency_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Generated: report/latency_distribution.png")

if __name__ == "__main__":
    generate_plots()
