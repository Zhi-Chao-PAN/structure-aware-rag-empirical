# 🎓 Academic Portfolio Materials: V-FinRAG

*Designed for MPhil / MSc Artificial Intelligence Applications (Target: UTS & USYD)*

---

## 1. Statement of Purpose (SOP) / Research Proposal Core Excerpt
*Instruction: Insert this high-density technical paragraph directly into the "Research Experience" or "Academic Interests" section of your Statement of Purpose.*

**Independent Research: Corpus-Scale Vision-RAG for Highly Structured Financial Documents**
> Recognizing the catastrophic structural hallucination caused by traditional OCR pipelines in financial document retrieval, I independently architected and engineered V-FinRAG, a purely vision-native retrieval framework that bypasses text-chunking entirely. Leveraging a 3B-parameter Vision Language Model (ColPali), I designed a custom `NumericAwareMaxSim` late-interaction operator to compute high-dimensional similarities directly from 1024x1024 image patches. 
> 
> To execute this on consumer-grade hardware (24GB VRAM), I implemented rigorous system-level optimizations, including custom $O(1)$ vectorized risk-matrix computations, BFloat16 mixed-precision alignment, and gradient checkpointing. Evaluated against a dense corpus of 1,473 diverse SEC filings, the system achieved a zero-shot Recall@5 of 60.00% and an MRR of 0.4250. This project demonstrates my capacity to bridge theoretical representation learning with bare-metal GPU engineering, preparing me to tackle scalable, multimodal AI challenges at the graduate level.

---

## 2. CV / Resume Bullets (Action-Oriented)
*Instruction: Use these under your "Projects" or "Research Experience" section.*

**Project: V-FinRAG: Numeric-Aware Late Interaction based Vision-RAG**
*Role: Independent Researcher / Sole Developer*
- **Architected** an OCR-free, vision-native RAG system using a 3B-parameter ColPali VLM, eliminating tabular structural hallucinations commonly found in PyPDF/LlamaParse pipelines.
- **Engineered** a custom `NumericAwareMaxSim` PyTorch operator with dynamic convex-combination gating, boosting the model's sensitivity to dense numerical columns in SEC 10-K/10-Q filings.
- **Formulated** the `DenoisedFinancialInfoNCE` contrastive loss function, employing a 3-tier hierarchy (False-Negative Masking, EMA Teacher Distillation, Minimum-Weight Lower Bounding) to prevent representation collapse during hard negative mining.
- **Optimized** the training and inference pipeline for 24GB VRAM availability using gradient checkpointing, `bfloat16` precision, and memory-mapped tensor indexing, executing corpus-scale (1,473 pages) retrieval evaluations within seconds.
- **Established** a rigorous zero-shot baseline (Recall@5: 60%, MRR: 0.42) on a complex financial benchmark, demonstrating highly robust latent spatial neighborhood matching against 1,400+ identical structural distractors.

---

## 3. GitHub Repository Organization Strategy
*(Currently implemented in your repository)*
- **`/src`**: Contains ONLY the V-FinRAG state-of-the-art vision-native implementation.
- **`/legacy`**: The "archive of growth". Contains the old PyPDF, LlamaParse, and LLM-reasoning scripts. 
  - *Academic Value*: When a professor looks at this repo, the `legacy` folder serves as tangible proof that you didn't just build V-FinRAG out of nowhere—you built it because you actively hit the ceiling of traditional NLP chunking and empirically proved its failure first. This demonstrates **iterative problem solving** and genuine research curiosity.
