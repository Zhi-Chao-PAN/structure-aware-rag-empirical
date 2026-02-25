"""
空间掩码计算模块 (Spatial Mask)

核心职责:
  1. Fit Resize 坐标变换 (保持 PDF 长宽比, 短边 Padding)
  2. IoPA (Intersection over Patch Area) 连续软掩码计算
  3. Padding 区域掩码生成 (用于 Attention 屏蔽)

数学基础:
  M_soft^{(r,c)} = (Δx_overlap · Δy_overlap) / P²
  其中 Δx_overlap = max(0, min(x'_max, px_max) - max(x'_min, px_min))

依赖: 仅依赖 torch, 无外部 ML 框架依赖
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class FitResizeInfo:
    """Fit Resize 变换的完整参数，用于坐标系转换和逆映射"""
    scale: float           # 缩放因子 s = min(W_img/W_page, H_img/H_page)
    pad_x: float           # x 方向 padding 偏移量 (左侧)
    pad_y: float           # y 方向 padding 偏移量 (顶部)
    img_size: int           # ViT 输入尺寸 (正方形边长)
    page_width: float       # 原始 PDF 页面宽度
    page_height: float      # 原始 PDF 页面高度


def compute_fit_resize_info(
    page_width: float,
    page_height: float,
    img_size: int = 1024,
) -> FitResizeInfo:
    """
    计算 Fit Resize (保持长宽比 + Padding) 的变换参数。

    Args:
        page_width: PDF 页面宽度 (PyMuPDF 单位: pt)
        page_height: PDF 页面高度 (PyMuPDF 单位: pt)
        img_size: ViT 输入正方形边长 (像素)

    Returns:
        FitResizeInfo: 包含 scale, pad_x, pad_y 等完整变换参数
    """
    scale = min(img_size / page_width, img_size / page_height)
    pad_x = (img_size - scale * page_width) / 2.0
    pad_y = (img_size - scale * page_height) / 2.0

    return FitResizeInfo(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        img_size=img_size,
        page_width=page_width,
        page_height=page_height,
    )


def pdf_bbox_to_pixel(
    bbox: tuple[float, float, float, float],
    info: FitResizeInfo,
) -> tuple[float, float, float, float]:
    """
    将 PDF 坐标系的 BBox 转换为 ViT 输入图像的像素坐标。

    Args:
        bbox: (x_min, y_min, x_max, y_max) PyMuPDF 坐标 (左上角原点)
        info: Fit Resize 变换参数

    Returns:
        (x'_min, y'_min, x'_max, y'_max) 像素坐标
    """
    x_min, y_min, x_max, y_max = bbox
    return (
        x_min * info.scale + info.pad_x,
        y_min * info.scale + info.pad_y,
        x_max * info.scale + info.pad_x,
        y_max * info.scale + info.pad_y,
    )


def compute_iopa_mask(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    img_size: int = 1024,
    patch_size: int = 14,
    has_cls: bool = True,
) -> torch.Tensor:
    """
    计算单个 BBox 的 IoPA (Intersection over Patch Area) 连续软掩码。

    核心公式:
        M_soft^{(r,c)} = max(0, min(x'_max, px_max) - max(x'_min, px_min))
                       × max(0, min(y'_max, py_max) - max(y'_min, py_min))
                       / P²

    Args:
        bbox: (x_min, y_min, x_max, y_max) PDF 坐标系
        page_width: PDF 页面宽度
        page_height: PDF 页面高度
        img_size: ViT 输入图像尺寸 (正方形)
        patch_size: Patch 边长 P
        has_cls: 是否有 [CLS] token (影响序列长度)

    Returns:
        mask: shape (L,), L = N_h*N_w [+1 if has_cls]
              值域 [0, 1], 表示每个 patch 被 BBox 覆盖的面积占比
    """
    N_h = img_size // patch_size
    N_w = img_size // patch_size
    P = patch_size

    # Step 1: PDF → 像素坐标
    info = compute_fit_resize_info(page_width, page_height, img_size)
    x_min_px, y_min_px, x_max_px, y_max_px = pdf_bbox_to_pixel(bbox, info)

    # Step 2: 向量化计算所有 patch 的 IoPA
    # 构建每个 patch 的边界坐标 (向量化, 无循环)
    cols = torch.arange(N_w, dtype=torch.float32)  # [N_w]
    rows = torch.arange(N_h, dtype=torch.float32)  # [N_h]

    # Patch 左边界和右边界
    px_min_x = cols * P          # [N_w]
    px_max_x = (cols + 1) * P    # [N_w]
    px_min_y = rows * P          # [N_h]
    px_max_y = (rows + 1) * P    # [N_h]

    # 计算 x 方向重叠 [N_w]
    overlap_x = torch.clamp(
        torch.minimum(torch.tensor(x_max_px), px_max_x) -
        torch.maximum(torch.tensor(x_min_px), px_min_x),
        min=0.0,
    )

    # 计算 y 方向重叠 [N_h]
    overlap_y = torch.clamp(
        torch.minimum(torch.tensor(y_max_px), px_max_y) -
        torch.maximum(torch.tensor(y_min_px), px_min_y),
        min=0.0,
    )

    # 外积得到 2D 面积重叠矩阵 [N_h, N_w]
    overlap_area = torch.outer(overlap_y, overlap_x)

    # IoPA: 面积占比
    iopa = overlap_area / (P * P)

    # Step 3: 展平为 1D patch 序列 (row-major)
    mask_flat = iopa.reshape(-1)  # [N_h * N_w]

    # 加入 [CLS] token 位置 (掩码值为 0)
    if has_cls:
        cls_pad = torch.zeros(1, dtype=torch.float32)
        mask_flat = torch.cat([cls_pad, mask_flat], dim=0)

    return mask_flat


def compute_multi_bbox_mask(
    bboxes: list[tuple[float, float, float, float]],
    page_width: float,
    page_height: float,
    img_size: int = 1024,
    patch_size: int = 14,
    has_cls: bool = True,
) -> torch.Tensor:
    """
    计算多个 BBox 的联合软掩码 (element-wise max)。
    用于算术型问题: 多个操作数的 BBox 合并为一个 GT mask。

    Args:
        bboxes: BBox 列表
        其他参数同 compute_iopa_mask

    Returns:
        mask: shape (L,), 每个 patch 取所有 BBox 中的最大 IoPA 值
    """
    if not bboxes:
        seq_len = (img_size // patch_size) ** 2 + (1 if has_cls else 0)
        return torch.zeros(seq_len, dtype=torch.float32)

    masks = []
    for bbox in bboxes:
        m = compute_iopa_mask(bbox, page_width, page_height,
                              img_size, patch_size, has_cls)
        masks.append(m)

    # Union: element-wise max (不是 sum, 避免重叠区域 > 1.0)
    stacked = torch.stack(masks, dim=0)  # [K, L]
    return stacked.max(dim=0).values     # [L]


def compute_padding_mask(
    page_width: float,
    page_height: float,
    img_size: int = 1024,
    patch_size: int = 14,
    has_cls: bool = True,
) -> torch.Tensor:
    """
    生成 Padding 区域的掩码: 有效内容区域 = 1, Padding 区域 = 0。
    用于在 Attention 计算中屏蔽 Padding patch。

    Args:
        page_width: PDF 页面宽度
        page_height: PDF 页面高度
        img_size: ViT 输入尺寸
        patch_size: Patch 尺寸
        has_cls: 是否有 [CLS] token

    Returns:
        mask: shape (L,), 1.0 表示有效内容, 0.0 表示 Padding
    """
    info = compute_fit_resize_info(page_width, page_height, img_size)
    N_h = img_size // patch_size
    N_w = img_size // patch_size
    P = patch_size

    # 有效内容在图像中的像素范围
    content_x_min = info.pad_x
    content_x_max = img_size - info.pad_x
    content_y_min = info.pad_y
    content_y_max = img_size - info.pad_y

    # 用 IoPA 的方式计算: 有效内容区域与每个 patch 的交叠面积占比
    # 只要 patch 的中心在有效区域内, 就视为有效 (阈值法更鲁棒)
    cols = torch.arange(N_w, dtype=torch.float32)
    rows = torch.arange(N_h, dtype=torch.float32)

    # Patch 中心坐标
    center_x = (cols + 0.5) * P  # [N_w]
    center_y = (rows + 0.5) * P  # [N_h]

    # 判断中心是否在有效区域内
    valid_x = (center_x >= content_x_min) & (center_x < content_x_max)  # [N_w]
    valid_y = (center_y >= content_y_min) & (center_y < content_y_max)  # [N_h]

    # 外积得到 2D 有效性矩阵
    valid_2d = torch.outer(valid_y.float(), valid_x.float())  # [N_h, N_w]
    mask = valid_2d.reshape(-1)  # [N_h * N_w]

    if has_cls:
        # [CLS] token 始终有效
        cls_valid = torch.ones(1, dtype=torch.float32)
        mask = torch.cat([cls_valid, mask], dim=0)

    return mask
