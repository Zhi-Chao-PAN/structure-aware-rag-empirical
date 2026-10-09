"""
多粒度评测指标体系 (Multi-Granularity Evaluation Metrics)

指标层级:
  Level 1: Page-Level   → Recall@K, nDCG@K, MRR
  Level 2: Region-Level → VGS-Hit, VGS-IoU, VGS-Corr → GRS
  Level 3: Adversarial  → ADR, AMR, NHR
"""

import torch
import numpy as np
from typing import Optional
from tqdm import tqdm


# =====================================================================
# Level 1: Page-Level Retrieval Metrics
# =====================================================================

def page_recall_at_k(
    retrieved_pages: list[list[str]],
    gt_pages: list[str],
    k: int = 5,
) -> float:
    """
    Recall@K: 正确页面出现在 top-K 中的比例。

    Args:
        retrieved_pages: [num_queries][K] 每个 query 的 top-K 检索页面 ID
        gt_pages: [num_queries] 每个 query 的正确页面 ID
    """
    hits = sum(1 for ret, gt in zip(retrieved_pages, gt_pages) if gt in ret[:k])
    return hits / max(len(gt_pages), 1)


def page_mrr(
    retrieved_pages: list[list[str]],
    gt_pages: list[str],
) -> float:
    """MRR: 正确页面首次出现排名的倒数的均值。"""
    reciprocal_ranks = []
    for ret, gt in zip(retrieved_pages, gt_pages):
        if gt in ret:
            rank = ret.index(gt) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks).item() if reciprocal_ranks else 0.0


def page_ndcg_at_k(
    retrieved_pages: list[list[str]],
    gt_pages: list[str],
    k: int = 5,
) -> float:
    """nDCG@K: 标准排序质量指标 (二值相关性)。"""
    ndcg_scores = []
    for ret, gt in zip(retrieved_pages, gt_pages):
        dcg = 0.0
        for rank, page in enumerate(ret[:k]):
            if page == gt:
                # 只有第 rank 个位置命中了，所以我们加上它带来的收益。二值相关性：rel = 1
                dcg += 1.0 / np.log2(rank + 2)  
                break # 既然每个 Query 只有一个正确 gt，找到就不用继续找了
        
        # 对于单一 GT 对应的问答情况，Ideal DCG 就是正确答案排在第一位的情况：即 np.log2(1+1) = 1.0
        idcg = 1.0 
        
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcg_scores).item() if ndcg_scores else 0.0


# =====================================================================
# Level 2: Region-Level Visual Grounding Metrics
# =====================================================================

def visual_grounding_score(
    pred_patch_weights: torch.Tensor,    # [num_queries, N_patches]
    gt_soft_masks: torch.Tensor,         # [num_queries, N_patches]
    top_k_patches: int = 10,
    iopa_threshold: float = 0.1,
) -> dict[str, float]:
    """
    视觉定位评分 (VGS): 模型是否"看"对了地方？

    三个子指标:
      VGS-Hit:  top-K 注意力 patch 中命中 GT 区域的比例
      VGS-IoU:  预测注意力区域与 GT 区域的 soft IoU
      VGS-Corr: 预测注意力分布与 GT 软掩码的 Pearson 相关系数

    Args:
        pred_patch_weights: [Q, L] 模型输出的 patch 注意力
        gt_soft_masks: [Q, L] IoPA GT 软掩码
        top_k_patches: VGS-Hit 的 top-K 值
        iopa_threshold: IoPA > 此值视为 GT 活跃 patch
    """
    Q = pred_patch_weights.shape[0]

    # --- VGS-Hit ---
    _, pred_top_idx = pred_patch_weights.topk(top_k_patches, dim=-1)  # [Q, K]
    gt_active = (gt_soft_masks > iopa_threshold).float()

    hits = 0.0
    for i in range(Q):
        pred_set = set(pred_top_idx[i].tolist())
        gt_set = set(torch.where(gt_active[i] > 0)[0].tolist())
        if gt_set:
            hits += len(pred_set & gt_set) / len(gt_set)
    vgs_hit = hits / max(Q, 1)

    # --- VGS-IoU ---
    pred_norm = pred_patch_weights - pred_patch_weights.min(dim=-1, keepdim=True).values
    pred_max = pred_norm.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    pred_norm = pred_norm / pred_max

    intersection = (pred_norm * gt_soft_masks).sum(dim=-1)
    union = (pred_norm + gt_soft_masks - pred_norm * gt_soft_masks).sum(dim=-1)
    vgs_iou = (intersection / union.clamp(min=1e-8)).mean().item()

    # --- VGS-Corr ---
    pred_c = pred_norm - pred_norm.mean(dim=-1, keepdim=True)
    gt_c = gt_soft_masks - gt_soft_masks.mean(dim=-1, keepdim=True)
    num = (pred_c * gt_c).sum(dim=-1)
    denom = (pred_c.norm(dim=-1) * gt_c.norm(dim=-1)).clamp(min=1e-8)
    vgs_corr = (num / denom).mean().item()

    return {
        f"VGS-Hit@{top_k_patches}": vgs_hit,
        "VGS-IoU": vgs_iou,
        "VGS-Corr": vgs_corr,
    }


def grounded_retrieval_score(
    page_recall_1: float,
    vgs_iou: float,
    beta: float = 0.5,
) -> float:
    """
    GRS: 综合页面检索正确性与视觉定位精度的单一标量。

    GRS = Page_Recall@1^(1-β) × VGS-IoU^β
    """
    return (page_recall_1 ** (1 - beta)) * (vgs_iou ** beta)


# =====================================================================
# Level 3: Adversarial Discrimination Metrics
# =====================================================================

def adversarial_discrimination_rate(
    scores_correct: torch.Tensor,       # [N_probes]
    scores_adversarial: torch.Tensor,   # [N_probes]
) -> dict[str, float]:
    """
    对抗区分率 (ADR): 模型能否在对抗对中选出正确页面。

    ADR: 硬判别准确率
    AMR: 平均分数差距 / 平均分数绝对值 (信心度)
    """
    correct_wins = (scores_correct > scores_adversarial).float()
    adr = correct_wins.mean().item()

    margin = scores_correct - scores_adversarial
    avg_score = (scores_correct.abs() + scores_adversarial.abs()) / 2
    amr = (margin / avg_score.clamp(min=1e-8)).mean().item()

    return {"ADR": adr, "AMR": amr}


def numeric_hallucination_rate(
    pred_patch_weights: torch.Tensor,    # [N, M]
    correct_number_mask: torch.Tensor,   # [N, M]
    wrong_number_masks: torch.Tensor,    # [N, M]
    top_k: int = 10,
) -> float:
    """
    数值幻觉率 (NHR): 模型注意力是否聚焦在错误的数字上。

    NHR = (top-K 中与错误数字重叠 > 与正确数字重叠的样本比例)
    NHR 越低越好, 理想 < 5%。
    """
    _, top_idx = pred_patch_weights.topk(top_k, dim=-1)  # [N, K]

    hallucination = 0
    total = pred_patch_weights.shape[0]

    for i in range(total):
        top_set = set(top_idx[i].tolist())
        correct_set = set(torch.where(correct_number_mask[i] > 0)[0].tolist())
        wrong_set = set(torch.where(wrong_number_masks[i] > 0)[0].tolist())

        correct_overlap = len(top_set & correct_set)
        wrong_overlap = len(top_set & wrong_set)

        if wrong_overlap > correct_overlap:
            hallucination += 1

    return hallucination / max(total, 1)


# =====================================================================
# Main Compute Execution Wrapper
# =====================================================================

def compute_metrics(model, test_loader, all_doc_embs, device, corpus_page_ids=None, use_maxsim=True):
    """
    通过流式分块前向计算查询特征，并在全量文档索引池中完成 Recall/nDCG 计算。
    
    Args:
        model: VisionRetriever
        test_loader: 包含 query 和 gt doc_id 的 DataLoader
        all_doc_embs: [Total_Docs, N, D] 的预计算文档 Embeddings
        device: 计算设备
        corpus_page_ids: [Total_Docs] 的 page_id 列表，对应 all_doc_embs 的顺序。
                         如果为 None，则假设 test_dataset 的顺序即为 doc 顺序。
        use_maxsim: 是否使用模型的 NumericAwareMaxSim (默认 True)。
                    False 时退化为 vanilla MaxSim (用于消融对比)。
    """
    print(f"[METRICS] 开始提取 Query 特征...")
    all_queries_embs = []
    all_gates = []
    gt_pages = []
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in tqdm(test_loader, desc="Encoding Queries"):
                queries = batch['query']
                doc_ids = batch['doc_id'] 
                
                query_emb, gate = model.encode_query(queries)
                query_emb = query_emb.to(device)
                
                all_queries_embs.append(query_emb)
                all_gates.append(gate)
                gt_pages.extend(doc_ids)
                
    print(f"[METRICS] 执行 Late Interaction (MaxSim)...")
    Total_Docs = all_doc_embs.shape[0]
    chunk_size = 200
    
    all_final_scores = []
    
    for qi, query_batch_emb in enumerate(tqdm(all_queries_embs, desc="Computing Scores")):
        batch_scores = []
        gate = all_gates[qi]
        
        for i in range(0, Total_Docs, chunk_size):
            end = min(i + chunk_size, Total_Docs)
            doc_chunk = all_doc_embs[i:end].to(device, non_blocking=True)
            
            if use_maxsim and hasattr(model, 'maxsim'):
                # 使用 NumericAwareMaxSim — 需要构造 dummy soft_mask
                B_q = query_batch_emb.shape[0]
                B_d = doc_chunk.shape[0]
                M = doc_chunk.shape[1]
                
                # 评测时无 IoPA mask，用全 1（等价于所有 patch 都是数字区域）
                dummy_mask = torch.ones(B_d, M, device=device)
                dummy_pad = torch.ones(B_d, M, device=device)
                
                # 逐 query batch 与 doc chunk 计算
                # 扩维做全对: E_q [B_q,1,N,D] x E_d [1,B_d,D,M] -> [B_q,B_d,N,M]
                from torch.nn.functional import normalize
                E_q_norm = normalize(query_batch_emb, p=2, dim=-1)
                E_d_norm = normalize(doc_chunk, p=2, dim=-1)
                
                sim = torch.matmul(
                    E_q_norm.unsqueeze(1),
                    E_d_norm.unsqueeze(0).transpose(-1, -2)
                )  # [B_q, B_d, N, M]
                
                S_std = sim.max(dim=-1).values  # [B_q, B_d, N]
                
                M_soft_exp = dummy_mask.unsqueeze(0).unsqueeze(2)  # [1, B_d, 1, M]
                sim_weighted = sim * M_soft_exp
                S_num = sim_weighted.max(dim=-1).values  # [B_q, B_d, N]
                
                gate_exp = gate.unsqueeze(1)  # [B_q, 1, N]
                gamma = model.maxsim.gamma
                S_token = (1.0 - gate_exp) * S_std + gate_exp * (S_num + gamma)
                chunk_scores = S_token.mean(dim=-1)  # [B_q, B_d]
            else:
                # Vanilla MaxSim fallback
                sim_matrix = torch.einsum('bqd, cnd -> bcqn', query_batch_emb, doc_chunk)
                max_sim_over_n, _ = sim_matrix.max(dim=-1)
                chunk_scores = max_sim_over_n.mean(dim=-1)
            
            batch_scores.append(chunk_scores.cpu())
            del doc_chunk
            
        all_final_scores.append(torch.cat(batch_scores, dim=1))
    
    final_scores = torch.cat(all_final_scores, dim=0)
    
    top_k = 5
    _, top_indices = final_scores.topk(min(top_k, Total_Docs), dim=1)
    
    if corpus_page_ids is not None:
        retrieved_pages = [[corpus_page_ids[idx] for idx in row.tolist()] for row in top_indices]
    else:
        dataset_doc_ids = [item['page_id'] for item in test_loader.dataset.data]
        retrieved_pages = [[dataset_doc_ids[idx] for idx in row.tolist()] for row in top_indices]
    
    print(f"[METRICS] 计算最终指标...")
    recall_1 = page_recall_at_k(retrieved_pages, gt_pages, k=1)
    recall_5 = page_recall_at_k(retrieved_pages, gt_pages, k=5)
    mrr = page_mrr(retrieved_pages, gt_pages)
    ndcg_5 = page_ndcg_at_k(retrieved_pages, gt_pages, k=5)
    
    detailed_results = []
    qs = [item['query'] for item in test_loader.dataset.data]
    for i in range(len(gt_pages)):
        gt = gt_pages[i]
        ret = retrieved_pages[i]
        r1 = 1 if gt in ret[:1] else 0
        r5 = 1 if gt in ret[:5] else 0
        dcg = 0.0
        for rank, page in enumerate(ret[:5]):
            if page == gt:
                dcg += 1.0 / np.log2(rank + 2)
                break
        ind_ndcg5 = dcg / 1.0 if dcg > 0 else 0.0
        
        detailed_results.append({
            "query": qs[i],
            "gt_page": gt,
            "retrieved_top5": ret[:5],
            "Hit@1": r1,
            "Hit@5": r5,
            "nDCG@5": round(ind_ndcg5, 4)
        })
    
    return {
        "Recall@1": list(map(lambda x: round(x, 4), [recall_1]))[0],
        "Recall@5": list(map(lambda x: round(x, 4), [recall_5]))[0],
        "MRR": list(map(lambda x: round(x, 4), [mrr]))[0],
        "nDCG@5": list(map(lambda x: round(x, 4), [ndcg_5]))[0],
        "details": detailed_results
    }
