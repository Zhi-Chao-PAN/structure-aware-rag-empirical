import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import textwrap

def visualize_hard_negatives(eval_results_path="eval_results.json", image_dir="data/vision_rag/page_images", output_dir="report/hard_negatives"):
    if not os.path.exists(eval_results_path):
        print(f"[ERROR] Evaluation results file not found at {eval_results_path}.")
        print("Please ensure `eval.py` has been completely run and saved the JSON output.")
        return

    with open(eval_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    details = data.get("details", [])
    if not details:
        print("[ERROR] No 'details' field found in eval_results.json. Was the script run with the latest metrics.py?")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    analyzed_count = 0
    for idx, item in enumerate(details):
        query = item.get("query", "Unknown Query")
        gt_page = item.get("gt_page")
        retrieved_top1 = item.get("retrieved_top5", [None])[0]
        
        # We only care about Hard Negatives (Recall@1 == False)
        if gt_page == retrieved_top1:
            continue
            
        print(f"[ANALYZING] Query: {query[:50]}... | GT: {gt_page} | Pred: {retrieved_top1}")
        
        gt_img_path = Path(image_dir) / f"{gt_page}.jpg"
        pred_img_path = Path(image_dir) / f"{retrieved_top1}.jpg"
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle(f"Hard Negative Analysis\nQuery: {textwrap.fill(query, width=90)}", fontsize=16, fontweight='bold', y=0.98)
        
        # Ground Truth
        if gt_img_path.exists():
            axes[0].imshow(Image.open(gt_img_path))
            axes[0].set_title(f"[GROUND TRUTH]\nPage ID: {gt_page}", color='green', fontsize=14, fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, 'Image Not Found', ha='center', va='center', fontsize=20)
            axes[0].set_title(f"[GROUND TRUTH] (Missing)\nPage ID: {gt_page}", color='red')
        axes[0].axis('off')
        
        # Retrieval Failure (Top-1)
        if pred_img_path.exists() and retrieved_top1 is not None:
            axes[1].imshow(Image.open(pred_img_path))
            axes[1].set_title(f"[TOP-1 FAILURE]\nPage ID: {retrieved_top1}", color='red', fontsize=14, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'Image Not Found', ha='center', va='center', fontsize=20)
            axes[1].set_title(f"[TOP-1 FAILURE] (Missing)\nPage ID: {retrieved_top1}", color='red')
        axes[1].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        safe_query = "".join([c if c.isalnum() else "_" for c in query[:30]])
        out_file = Path(output_dir) / f"fail_{idx}_{safe_query}.png"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        analyzed_count += 1
        
    print(f"\n[DONE] Rendered {analyzed_count} Hard Negative comparisons to {output_dir}/")
    print("Please review these images to diagnose the structural hallucinations (e.g., Twin Documents vs. Numeric Density).")

if __name__ == "__main__":
    visualize_hard_negatives()
