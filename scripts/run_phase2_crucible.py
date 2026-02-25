"""
Phase 2: The Micro-Universe Algorithm Meat Grinder
Task 06: Zero-Shot Baseline (The Forward Pass Crucible)
Task 07: Query Gate Activation Profiling
Task 08: 2-Sample Overfit Sanity Check

Hardware constraint: 8GB VRAM RTX 4060.
ColPali bfloat16 loaded in 6GB. Using torch.no_grad() for Task 06-07.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import json
import logging
from typing import Dict, Any

from src.vision_rag.model.vision_retriever import VisionRetriever
from src.vision_rag.eval.metrics import visual_grounding_score
from src.vision_rag.eval.attention_extract import extract_patch_attention
from src.vision_rag.loss.denoised_infonce import DenoisedFinancialInfoNCE

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 1024
PATCH_SIZE = 14
HAS_CLS = True
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) * (IMG_SIZE // PATCH_SIZE)
if HAS_CLS:
    NUM_PATCHES += 1  # 5329 + 1 = 5330

def load_oracle_qa(path: str = "data/vision_rag/oracle_qa.json") -> list[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_image(path: str) -> Image.Image:
    # Ensure standard RGB loading to send to PaliGemma
    return Image.open(path).convert("RGB")

def print_gate_profiling(query: str, tokens: list[str], gate_values: torch.Tensor):
    """Task 07: Gate Activation Profiling - Visualization"""
    print(f"\n[Task 07: Query Gate Profiling]")
    print(f"Query: {query}")
    print("-" * 60)
    
    # Exclude padding (usually <pad> or </s> tokens at the end)
    max_len = min(len(tokens), gate_values.shape[0])
    
    # Build formatted visual output
    out_tokens = []
    out_gates = []
    for i in range(max_len):
        tok = tokens[i].replace(" ", " ") # Replace sentencepiece space indicator
        g_val = gate_values[i].item()
        
        # Color coding in console (optional, using simple text distinction here)
        marker = "🔥" if g_val > 0.8 else "🔹" if g_val > 0.2 else "⚪"
        out_tokens.append(f"{tok:<12}")
        out_gates.append(f"{g_val:.3f} {marker}  ")
        
    print("".join(out_tokens[:15]))  # Print first 15 tokens
    print("".join(out_gates[:15]))
    print("-" * 60 + "\n")

def task_06_07_zero_shot_and_gate(retriever: VisionRetriever, data: list[Dict[str, Any]], device: str):
    """Run Task 06 (Zero-Shot VGS) and Task 07 (Gate Profiling)"""
    logger.info("=" * 70)
    logger.info("🗡️ TASK 06: The Zero-Shot Baseline (Forward Pass Crucible)")
    logger.info("🛡️ TASK 07: The Gate Activation Profiling")
    logger.info("=" * 70)
    
    retriever.eval()
    tokenizer = retriever.backbone.processor.tokenizer
    
    total_queries = len(data)
    vgs_sum_iou = 0.0
    vgs_sum_hit = 0.0
    
    with torch.no_grad():
        for i, item in enumerate(data):
            query_text = item["query"]
            img_path = f"data/vision_rag/page_images/{item['page_id']}.jpg"
            gt_mask_active_indices = item["reverse_map"]["activated_patch_indices"]
            
            # Reconstruction of boolean GT mask for VGS evaluation
            # Oracle QA was generated at 1024x1024 resolution (73x73 grid = 5329 patches)
            # ColPali-v1.2 uses 448x448 natively (32x32 grid = 1024 patches)
            
            # Extract 2D grid coordinates from JSON
            activated_2d = item["reverse_map"]["activated_patch_2d"] # List of [y, x] in 73x73 grid
            
            # Map 73x73 coordinates to 32x32 grid
            mapped_indices = []
            for y_73, x_73 in activated_2d:
                y_32 = int(round(y_73 * (32.0 / 73.0)))
                x_32 = int(round(x_73 * (32.0 / 73.0)))
                # clamp to valid range
                y_32 = min(max(y_32, 0), 31)
                x_32 = min(max(x_32, 0), 31)
                mapped_indices.append(y_32 * 32 + x_32)
                
            mapped_indices = list(set(mapped_indices)) # unique indices
            
            # ColPali patch output shape
            img = _load_image(img_path)
            E_d = retriever.encode_document([img]).to(device) # [1, Mlen, D]
            M_len = E_d.shape[1]
            
            # Typically 1024 image patches + some special tokens
            colpali_num_patches = 1024
            
            gt_soft_mask = torch.zeros(colpali_num_patches, device=device)
            if len(mapped_indices) > 0:
                gt_soft_mask[mapped_indices] = 1.0
                
            # Place the 1024 patches effectively where they belong in the full sequence
            eval_mask = torch.zeros(1, M_len, device=device)
            eval_mask[0, :colpali_num_patches] = gt_soft_mask

            # Query encoding & Gate
            E_q, gate = retriever.encode_query([query_text]) # [B, N, D], [B, N]
            E_q = E_q.to(device)
            gate = gate.to(device)
            
            # Extract Attention
            patch_attention = extract_patch_attention(E_q, E_d, gate) # [M]
            patch_attention = patch_attention.unsqueeze(0) # [1, M]
            
            # Evaluation
            vgs_res = visual_grounding_score(
                pred_patch_weights=patch_attention,
                gt_soft_masks=eval_mask,
                top_k_patches=10,
                iopa_threshold=0.5
            )
            vgs_sum_hit += vgs_res["VGS-Hit@10"]
            vgs_sum_iou += vgs_res["VGS-IoU"]
            
            # Print gate profiling for the first 2 queries
            if i < 2:
                inputs = retriever.backbone.processor.process_queries([query_text])
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                print_gate_profiling(query_text, tokens, gate[0])
            
            logger.info(f"  [{i+1}/{total_queries}] Zero-Shot VGS-IoU: {vgs_res['VGS-IoU']:.4f}, Hit@10: {vgs_res['VGS-Hit@10']:.4f}")

        logger.info("-" * 40)
        logger.info(f"  Final Zero-Shot Average VGS-IoU:    {vgs_sum_iou / total_queries:.4f}")
        logger.info(f"  Final Zero-Shot Average VGS-Hit@10: {vgs_sum_hit / total_queries:.4f}")
        logger.info("-" * 40 + "\n")

def task_08_overfit_sanity_check(retriever: VisionRetriever, data: list[Dict[str, Any]], device: str):
    """Run Task 08 - The 2-Sample Overfit Sanity Check"""
    logger.info("=" * 70)
    logger.info("☠️ TASK 08: The 2-Sample Overfit Sanity Check (Close Graph Eval)")
    logger.info("=" * 70)
    
    # 保证选择不同页面的样本来避免预训练模型的数学崩溃
    queries = [data[0]["query"], data[2]["query"]]
    img_paths = [
        f"data/vision_rag/page_images/{data[0]['page_id']}.jpg",
        f"data/vision_rag/page_images/{data[2]['page_id']}.jpg"
    ]
    
    meta_q = [{"query_id": "q_0", "ticker": data[0]["page_id"].split("_")[0], "industry": "mock"}, 
              {"query_id": "q_2", "ticker": data[2]["page_id"].split("_")[0], "industry": "mock"}]
    meta_d = [{"page_id": "p_0", "ticker": data[0]["page_id"].split("_")[0], "fiscal_year": 2022, "quarter": "Q1", "report_type": "10-Q", "industry": "mock", "content_type": "text"}, 
              {"page_id": "p_2", "ticker": data[2]["page_id"].split("_")[0], "fiscal_year": 2022, "quarter": "Q1", "report_type": "10-Q", "industry": "mock", "content_type": "text"}]
    
    # Ensure backbone (3B params) is completely frozen
    for p in retriever.backbone.parameters():
        p.requires_grad = False
        
    # We unfreeze OUR custom projection and MaxSim
    if hasattr(retriever.backbone, "custom_proj"):
        for p in retriever.backbone.custom_proj.parameters():
            p.requires_grad = True
            
    for p in retriever.maxsim.parameters():
        p.requires_grad = True
        
    # Check what requires grad
    trainable_params = [p for p in retriever.parameters() if p.requires_grad]
    logger.info(f"  Trainable params for overfit: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    loss_fn = DenoisedFinancialInfoNCE(
        tau_init=0.05,
        learn_tau=True,
        lambda_meta=0.7  # Restore metadata risk matrix
    ).to(device)
    
    retriever.train()
    
    epochs = 100
    
    logger.info("  Starting 2-sample aggressive overfit loop...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Load batch inside loop to save memory (although 2 images is small)
        imgs = [_load_image(p) for p in img_paths]
        
        # Forward pass (all_pairs = True for contrastive in-batch)
        # Note: in train mode we don't use no_grad, but backbone uses internally if we wrapped process properly.
        # Actually our custom encode_image uses with torch.no_grad() for the PaliGemma backbone automatically!
        # This is perfect. It only computes grad for `custom_proj`.
        
        # Soft mask dummy for shape [B, M]
        # In reality, soft_mask is used for spatial regularizer, but here we just test the InfoNCE loss
        E_d = retriever.encode_document(imgs).to(device)  # [2, M, 128]
        E_q, gate = retriever.encode_query(queries)       # [2, N, 128]
        E_q, gate = E_q.to(device), gate.to(device)
        
        B, M, _ = E_d.shape
        dummy_soft_masks = torch.ones(B, M, device=device)
        
        scores, _ = retriever.maxsim(E_q, E_d, dummy_soft_masks, padding_mask=None, all_pairs=True)
        
        # Custom InfoNCE
        loss_out = loss_fn(scores=scores, meta_q=meta_q, meta_d=meta_d, lambda_self=0.1)
        loss = loss_out.loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            gate_val = gate.mean().item()
            proj_grad = 0.0
            if retriever.backbone.custom_proj.weight.grad is not None:
                proj_grad = retriever.backbone.custom_proj.weight.grad.norm().item()
            gate_grad = 0.0
            if retriever.maxsim.gate_proj.weight.grad is not None:
                gate_grad = retriever.maxsim.gate_proj.weight.grad.norm().item()

            logger.info(f"    Epoch [{epoch:02d}/{epochs}] Loss: {loss.item():.4f} "
                        f"| pos: {loss_out.pos_sim_mean:.4f} | neg: {loss_out.neg_sim_mean:.4f} "
                        f"| gate: {gate_val:.4f} | p_grad: {proj_grad:.4f} | g_grad: {gate_grad:.4f}")

    logger.info("  Overfit completed. If loss dropped near ~0 and pos_sim >> neg_sim, the graph is mathematically valid.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Device: {device}")
    
    logger.info("Initializing VisionRetriever (Loading ColPali-v1.2 in bfloat16)...")
    
    # In 8GB VRAM we must be very careful.
    retriever = VisionRetriever(
        use_mock=False,
        backbone_name="vidore/colpali-v1.2",
        freeze_backbone=True, # explicitly freeze
    )
    retriever = retriever.to(device)
    retriever = retriever.to(retriever.backbone.model.dtype)
    
    # Load data
    oracle_data = load_oracle_qa()
    
    # Run Tasks
    task_06_07_zero_shot_and_gate(retriever, oracle_data, device)
    
    # Free cache before task 08 just in case
    torch.cuda.empty_cache()
    
    task_08_overfit_sanity_check(retriever, oracle_data, device)
    
if __name__ == "__main__":
    main()
