# Vision-RAG: Structure-Aware Vision Late Interaction Retrieval
# 基于 Document-as-Image 范式的金融文档检索系统
"""
核心模块:
- model/: ViT backbone, Query Encoder, Numeric-Aware MaxSim 算子
- loss/: 三层去噪 InfoNCE + 课程学习调度
- data/: IoPA 空间掩码, 同心对抗采样器, 变长 Batch 整理
- eval/: 多粒度评测指标 (VGS-IoU, GRS, ADR, NHR)
"""
