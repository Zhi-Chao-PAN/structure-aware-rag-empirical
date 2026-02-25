"""
同心对抗批次采样器 (Concentric Adversarial Sampler)

设计原则:
  每个 Batch = K 个行业 × 2 clusters/行业 × cluster_size 个样本

  Cluster 内结构:
    Ring 0: Anchor (query + ground truth positive)
    Ring 1: 同公司 + 跨周期 (最难 → FN 高危 → Layer 1 重点去噪)
    Ring 2: 同行业 + 跨公司 (难 → 核心学习区)
    Ring 3: 跨行业 (隐式, 来自不同 Cluster → 提供 baseline 排斥力)

  课程学习维度:
    Easy (epoch 0~2):    K=8 industries → 每组 2 样本
    Balanced (epoch 3~6): K=4 → 每组 4 样本
    Hard (epoch 7+):     K=2 → 每组 8 样本
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import Sampler


@dataclass
class DocSample:
    """单个 (query, document) 样本的元数据"""
    idx: int                     # 在数据集中的全局索引
    ticker: str                  # 公司代号
    industry: str                # 行业
    fiscal_year: int             # 财年
    quarter: str                 # 季度
    report_type: str             # 报表类型
    content_type: str            # 内容类型
    bm25_hard_negatives: list[int] = field(default_factory=list)


class ConcentricAdversarialSampler(Sampler):
    """
    同心对抗批次采样器。

    Args:
        samples: 所有样本的元数据列表
        batch_size: 批次大小
        n_industries: 每 batch 行业数 (由 CurriculumScheduler 外部设置)
        clusters_per_industry: 每行业 cluster 数
        seed: 随机种子
    """

    def __init__(
        self,
        samples: list[DocSample],
        batch_size: int = 32,
        n_industries: int = 4,
        clusters_per_industry: int = 2,
        seed: int = 42,
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.n_industries = n_industries
        self.clusters_per_industry = clusters_per_industry
        self.seed = seed
        self.epoch = 0

        # 构建多级索引: industry → ticker → period → [sample_idx]
        self.industry_index: dict[str, dict[str, dict[str, list[int]]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.industry_to_tickers: dict[str, list[str]] = defaultdict(list)

        seen_tickers: dict[str, set] = defaultdict(set)
        for s in samples:
            period = f"{s.fiscal_year}{s.quarter}"
            self.industry_index[s.industry][s.ticker][period].append(s.idx)
            seen_tickers[s.industry].add(s.ticker)

        for ind, tickers in seen_tickers.items():
            self.industry_to_tickers[ind] = sorted(tickers)

        self.all_industries = sorted(self.industry_index.keys())

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def set_n_industries(self, n: int):
        """由外部 CurriculumScheduler 调用, 动态调整每 batch 行业数"""
        self.n_industries = n

    @property
    def cluster_size(self) -> int:
        n_clusters = self.n_industries * self.clusters_per_industry
        return max(2, self.batch_size // n_clusters)

    def _build_cluster(self, industry: str, rng: random.Random) -> list[int]:
        """构建一个对抗 Cluster: Ring 0 + Ring 1 + Ring 2 + BM25 补位"""
        tickers = self.industry_to_tickers.get(industry, [])
        target_size = self.cluster_size
        cluster = []

        if len(tickers) < 1:
            return cluster

        # Ring 0: Anchor
        anchor_ticker = rng.choice(tickers)
        periods = list(self.industry_index[industry][anchor_ticker].keys())
        if not periods:
            return cluster
        anchor_period = rng.choice(periods)
        candidates = self.industry_index[industry][anchor_ticker][anchor_period]
        anchor_idx = rng.choice(candidates)
        cluster.append(anchor_idx)

        # Ring 1: 同公司 + 不同周期 (优先时间最近)
        other_periods = [p for p in periods if p != anchor_period]
        if other_periods and len(cluster) < target_size:
            closest = min(other_periods,
                          key=lambda p: abs(self._period_ord(p) - self._period_ord(anchor_period)))
            r1_candidates = self.industry_index[industry][anchor_ticker][closest]
            cluster.append(rng.choice(r1_candidates))

        # Ring 2: 同行业 + 不同公司
        other_tickers = [t for t in tickers if t != anchor_ticker]
        rng.shuffle(other_tickers)
        for t in other_tickers:
            if len(cluster) >= target_size:
                break
            t_periods = list(self.industry_index[industry][t].keys())
            pick = anchor_period if anchor_period in t_periods else rng.choice(t_periods)
            cands = self.industry_index[industry][t][pick]
            idx = rng.choice(cands)
            if idx not in cluster:
                cluster.append(idx)

        # BM25 补位
        anchor_sample = self.samples[anchor_idx]
        while len(cluster) < target_size:
            if anchor_sample.bm25_hard_negatives:
                neg = rng.choice(anchor_sample.bm25_hard_negatives)
                if neg not in cluster:
                    cluster.append(neg)
                    continue
            # 兜底: 同行业随机
            all_idx = []
            for t_dict in self.industry_index[industry].values():
                for p_list in t_dict.values():
                    all_idx.extend(p_list)
            if all_idx:
                fb = rng.choice(all_idx)
                if fb not in cluster:
                    cluster.append(fb)
            else:
                break

        return cluster[:target_size]

    @staticmethod
    def _period_ord(period: str) -> int:
        try:
            year = int(period[:4])
            q = int(period[5]) if len(period) > 5 and period[5].isdigit() else 1
            return year * 4 + q
        except (ValueError, IndexError):
            return 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        n_batches = max(1, len(self.samples) // self.batch_size)

        for _ in range(n_batches):
            batch = []
            n_ind = min(self.n_industries, len(self.all_industries))
            selected = rng.sample(self.all_industries, n_ind) if n_ind <= len(self.all_industries) else self.all_industries[:]

            for industry in selected:
                for _ in range(self.clusters_per_industry):
                    cluster = self._build_cluster(industry, rng)
                    batch.extend(cluster)

            # 补齐到 batch_size
            while len(batch) < self.batch_size:
                batch.append(rng.randint(0, len(self.samples) - 1))

            yield batch[:self.batch_size]

    def __len__(self):
        return max(1, len(self.samples) // self.batch_size)
