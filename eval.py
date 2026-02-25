"""
Vision-RAG 评测入口 (Evaluation Entry Point)

支持三级评测:
  Level 1: Page Recall@K, nDCG@K, MRR
  Level 2: VGS-Hit, VGS-IoU, VGS-Corr → GRS
  Level 3: ADR, AMR, NHR (对抗评测, 需预构建 probe 集)

用法:
  python eval.py --config config/vision_rag/default.yaml --checkpoint checkpoints/vision_rag/epoch_14.pt --mock
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision_rag.model.vision_retriever import VisionRetriever
from vision_rag.eval.metrics import (
    page_recall_at_k,
    page_mrr,
    page_ndcg_at_k,
    visual_grounding_score,
    grounded_retrieval_score,
)
from vision_rag.eval.attention_extract import extract_patch_attention
from vision_rag.utils.config import load_config, VisionRAGConfig


def evaluate(config: VisionRAGConfig, args):
    """主评测函数"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评测设备: {device}")

    # ---- 模型加载 ----
    model = VisionRetriever(
        projection_dim=config.model.projection_dim,
        gamma=config.model.gamma,
        use_mock=args.mock,
    )

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[CKPT] 已加载: {args.checkpoint}")

    model = model.to(device)
    model.eval()

    # ---- 评测数据 ----
    if args.mock:
        print("[MOCK] 使用合成数据评测 (验证推理与指标计算流程)")
        results = run_mock_evaluation(model, config, device)
    else:
        print(f"[EVAL] 正在初始化真实财报测试集...")
        
        # 1. 初始化测试集 (这里采用内存映射与生成器模式，避免全量加载OOM)
        # 注意：真实 ColPaliBackbone 需要 PIL Image，所以 transform 定义为空或 None
        test_transform = None

        # 1. 加载测试集 (仅用于提取 Query)
        from vision_rag.data.test_dataset import FinancialTestDataset
        from vision_rag.data.corpus_dataset import FinancialCorpusDataset
        
        # 路径自适应逻辑
        data_dir = "/root/autodl-tmp/project/data/vision_rag/page_images" if os.path.exists("/root/autodl-tmp/project/data/vision_rag/page_images") else "data/vision_rag/page_images"
        
        # 优先寻找全量标注文件
        potential_full_anno = "/root/autodl-tmp/project/data/vision_rag/visfinqa/test.json"
        if not os.path.exists(potential_full_anno):
            potential_full_anno = "data/vision_rag/visfinqa/test.json"
        
        anno_file = potential_full_anno if os.path.exists(potential_full_anno) else "data/vision_rag/oracle_qa.json"
        print(f"[EVAL] 查询源: {anno_file}")

        test_dataset = FinancialTestDataset(data_dir=data_dir, annotation_file=anno_file)
        
        def test_collate(batch):
            return {
                'query': [item['query'] for item in batch],
                'doc_id': [item['doc_id'] for item in batch]
            }
        
        test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False, collate_fn=test_collate)

        # 2. 构建全量 Corpus 索引池 (1422+ 页)
        print(f"[EVAL] 正在为全量语料池提取特征: {data_dir}")
        corpus_dataset = FinancialCorpusDataset(data_dir=data_dir)
        
        def corpus_collate(batch):
            return {
                'image': [item['image'] for item in batch],
                'page_id': [item['page_id'] for item in batch]
            }
            
        corpus_loader = DataLoader(corpus_dataset, batch_size=config.data.batch_size, shuffle=False, collate_fn=corpus_collate)

        all_doc_embs = []
        all_corpus_page_ids = []
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for batch in tqdm(corpus_loader, desc="Encoding Corpus"):
                    doc_emb = model.encode_document(batch['image'])
                    all_doc_embs.append(doc_emb.cpu().pin_memory())
                    all_corpus_page_ids.extend(batch['page_id'])
        
        if len(all_doc_embs) > 0:
            all_doc_embs = torch.cat(all_doc_embs, dim=0).to(device)
            
            # 由于 Corpus 顺序可能与 Test Dataset 顺序不同，我们需要将 corpus_page_ids 传给 compute_metrics
            print(f"[EVAL] 开始在大海中检索 (Corpus Size: {len(all_corpus_page_ids)}, Queries: {len(test_dataset)})")
            
            from vision_rag.eval.metrics import compute_metrics
            results = compute_metrics(model, test_loader, all_doc_embs, device, corpus_page_ids=all_corpus_page_ids)
        else:
            results = {"Recall@1": 0.0, "Recall@5": 0.0, "MRR": 0.0, "nDCG@5": 0.0}

    # ---- 输出结果 ----
    print("\n" + "=" * 60)
    print("评测结果")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: [详见 {args.output or 'eval_results.json'}]")

    # 保存到 JSON
    out_path = Path(args.output) if args.output else Path("eval_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    return results


def run_mock_evaluation(
    model: VisionRetriever,
    config: VisionRAGConfig,
    device: torch.device,
) -> dict[str, float]:
    """用合成数据验证评测管线完整性"""
    B = 8
    M = 1030
    D = config.model.projection_dim

    with torch.no_grad():
        # 合成 query 和 doc
        images = torch.randn(B, 3, 1024, 1024, device=device)
        query_text = [f"What was revenue in Q{i+1}?" for i in range(B)]
        soft_masks = torch.rand(B, M, device=device)
        padding_mask = torch.ones(B, M, device=device)

        # 模型前向
        output = model(
            images=images,
            query_text=query_text,
            soft_masks=soft_masks,
            padding_mask=padding_mask,
            all_pairs=True,
        )

        scores = output["scores"]  # [B, B]

        # Level 1: Page-Level (模拟)
        _, top_indices = scores.topk(min(5, B), dim=-1)
        retrieved_pages = [[f"page_{j}" for j in row.tolist()] for row in top_indices]
        gt_pages = [f"page_{i}" for i in range(B)]

        recall_1 = page_recall_at_k(retrieved_pages, gt_pages, k=1)
        recall_5 = page_recall_at_k(retrieved_pages, gt_pages, k=5)
        mrr = page_mrr(retrieved_pages, gt_pages)
        ndcg_5 = page_ndcg_at_k(retrieved_pages, gt_pages, k=5)

        # Level 2: Region-Level
        pred_weights = torch.rand(B, M, device=device)
        gt_masks = soft_masks

        vgs = visual_grounding_score(
            pred_weights, gt_masks,
            top_k_patches=config.eval.top_k_patches,
            iopa_threshold=config.eval.vgs_iou_threshold,
        )

        grs = grounded_retrieval_score(
            recall_1, vgs["VGS-IoU"],
            beta=config.eval.grs_beta,
        )

    results = {
        "Recall@1": recall_1,
        "Recall@5": recall_5,
        "MRR": mrr,
        "nDCG@5": ndcg_5,
        **vgs,
        "GRS": grs,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Vision-RAG 评测")
    parser.add_argument("--config", type=str, default="config/vision_rag/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="使用合成数据")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args)


if __name__ == "__main__":
    main()
