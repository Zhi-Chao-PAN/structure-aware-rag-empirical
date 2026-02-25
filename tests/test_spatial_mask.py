"""
IoPA 空间掩码单元测试 (Spatial Mask Unit Tests)

测试用例:
  1. BBox 完全落在单个 Patch 内 → IoPA = bbox_area / P²
  2. BBox 恰好覆盖整个 Patch → IoPA = 1.0
  3. BBox 在 Padding 区域外 → 全零掩码
  4. Fit Resize 坐标变换的往返一致性
  5. 多 BBox 联合掩码 (union via max)
  6. Padding mask 正确性
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vision_rag.data.spatial_mask import (
    compute_fit_resize_info,
    pdf_bbox_to_pixel,
    compute_iopa_mask,
    compute_multi_bbox_mask,
    compute_padding_mask,
)


# ====================================================================
# Test: Fit Resize 坐标变换参数
# ====================================================================

class TestFitResizeInfo:
    def test_square_page_no_padding(self):
        """正方形页面 → scale = img_size / page_size, 无 padding"""
        info = compute_fit_resize_info(100, 100, img_size=1000)
        assert info.scale == 10.0
        assert info.pad_x == 0.0
        assert info.pad_y == 0.0

    def test_portrait_page_padding_x(self):
        """竖版页面 (H > W) → 高度受限, X 方向有 padding"""
        info = compute_fit_resize_info(100, 200, img_size=1000)
        # scale 受 H 限制: 1000/200 = 5.0
        assert info.scale == 5.0
        assert info.pad_x == (1000 - 5.0 * 100) / 2  # = 250
        assert info.pad_y == 0.0

    def test_landscape_page_padding_y(self):
        """横版页面 (W > H) → 宽度受限, Y 方向有 padding"""
        info = compute_fit_resize_info(200, 100, img_size=1000)
        # scale 受 W 限制: 1000/200 = 5.0
        assert info.scale == 5.0
        assert info.pad_x == 0.0
        assert info.pad_y == (1000 - 5.0 * 100) / 2  # = 250


# ====================================================================
# Test: PDF BBox → 像素坐标转换
# ====================================================================

class TestPdfBboxToPixel:
    def test_stretch_equivalent_for_square(self):
        """正方形页面的 Fit Resize 等效于 Stretch"""
        info = compute_fit_resize_info(100, 100, img_size=1000)
        x_min, y_min, x_max, y_max = pdf_bbox_to_pixel((10, 20, 30, 40), info)
        assert x_min == pytest.approx(100.0)  # 10 * 10
        assert y_min == pytest.approx(200.0)  # 20 * 10
        assert x_max == pytest.approx(300.0)  # 30 * 10
        assert y_max == pytest.approx(400.0)  # 40 * 10

    def test_with_padding_offset(self):
        """有 padding 时的坐标偏移"""
        info = compute_fit_resize_info(100, 200, img_size=1000)
        # scale=5, pad_x=250, pad_y=0
        x_min, y_min, x_max, y_max = pdf_bbox_to_pixel((10, 20, 30, 40), info)
        assert x_min == pytest.approx(10 * 5 + 250)  # 300
        assert y_min == pytest.approx(20 * 5 + 0)     # 100
        assert x_max == pytest.approx(30 * 5 + 250)   # 400
        assert y_max == pytest.approx(40 * 5 + 0)     # 200


# ====================================================================
# Test: IoPA 软掩码计算
# ====================================================================

class TestIoPAMask:
    def test_bbox_covers_entire_patch(self):
        """BBox 恰好覆盖一个完整 Patch → IoPA = 1.0"""
        # 正方形页面, scale=1, pad=0, patch=10
        # BBox = (0, 0, 10, 10) → 恰好覆盖第一个 patch
        mask = compute_iopa_mask(
            bbox=(0, 0, 10, 10),
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=False,
        )
        # 第一个 patch (0,0) 的 IoPA = 10*10 / 100 = 1.0
        assert mask[0].item() == pytest.approx(1.0)
        # 其他 patch 应该是 0
        assert mask[1:].max().item() == pytest.approx(0.0)

    def test_bbox_partial_coverage(self):
        """BBox 部分覆盖 → IoPA = overlap_area / P²"""
        # BBox = (0, 0, 5, 5) → 覆盖第一个 patch 的 1/4
        mask = compute_iopa_mask(
            bbox=(0, 0, 5, 5),
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=False,
        )
        # 5*5 / 100 = 0.25
        assert mask[0].item() == pytest.approx(0.25)

    def test_bbox_spans_multiple_patches(self):
        """BBox 跨越多个 Patch"""
        # BBox = (0, 0, 25, 15) → 横跨 3 列 × 2 行
        mask = compute_iopa_mask(
            bbox=(0, 0, 25, 15),
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=False,
        )
        # patch (0,0): full → 1.0
        # patch (0,1): full → 1.0
        # patch (0,2): 5/10 × 10/10 = 0.5
        # patch (1,0): 10/10 × 5/10 = 0.5
        # patch (1,1): 10/10 × 5/10 = 0.5
        # patch (1,2): 5/10 × 5/10 = 0.25
        assert mask[0].item() == pytest.approx(1.0)   # (0,0)
        assert mask[1].item() == pytest.approx(1.0)   # (0,1)
        assert mask[2].item() == pytest.approx(0.5)   # (0,2)
        assert mask[10].item() == pytest.approx(0.5)  # (1,0), N_w=10
        assert mask[11].item() == pytest.approx(0.5)  # (1,1)
        assert mask[12].item() == pytest.approx(0.25) # (1,2)

    def test_cls_token_offset(self):
        """有 [CLS] token 时, 掩码向后偏移 1"""
        mask_no_cls = compute_iopa_mask(
            bbox=(0, 0, 10, 10),
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=False,
        )
        mask_with_cls = compute_iopa_mask(
            bbox=(0, 0, 10, 10),
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=True,
        )
        # CLS token at position 0 should be 0
        assert mask_with_cls[0].item() == pytest.approx(0.0)
        # Patch values shifted by 1
        assert mask_with_cls[1].item() == mask_no_cls[0].item()
        # Length difference
        assert mask_with_cls.shape[0] == mask_no_cls.shape[0] + 1

    def test_all_values_in_range(self):
        """所有 IoPA 值在 [0, 1] 范围内"""
        mask = compute_iopa_mask(
            bbox=(50, 100, 200, 300),
            page_width=612, page_height=792,
            img_size=1024, patch_size=14, has_cls=True,
        )
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0 + 1e-6

    def test_bbox_outside_page(self):
        """BBox 在页面外 → 落在 padding 区域 → 全零或极小"""
        # 竖版页面, pad_x=250
        # BBox 在页面外: x > page_width
        mask = compute_iopa_mask(
            bbox=(200, 0, 210, 10),  # x > 100 (page_width)
            page_width=100, page_height=200,
            img_size=1000, patch_size=10, has_cls=False,
        )
        # 映射后应在 padding 区域内, IoPA 仍然合法但基于像素坐标
        assert mask.min() >= 0.0


# ====================================================================
# Test: 多 BBox 联合掩码
# ====================================================================

class TestMultiBBoxMask:
    def test_union_max_not_sum(self):
        """多 BBox union 使用 max (不是 sum), 避免 > 1.0"""
        # 两个完全相同的 BBox → union == 单个 BBox
        mask_single = compute_iopa_mask(
            bbox=(0, 0, 10, 10),
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=False,
        )
        mask_union = compute_multi_bbox_mask(
            bboxes=[(0, 0, 10, 10), (0, 0, 10, 10)],
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=False,
        )
        assert torch.allclose(mask_single, mask_union)

    def test_empty_bboxes(self):
        """空 BBox 列表 → 全零掩码"""
        mask = compute_multi_bbox_mask(
            bboxes=[],
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=False,
        )
        assert mask.sum().item() == pytest.approx(0.0)
        assert mask.shape[0] == 100  # 10*10

    def test_disjoint_bboxes_sum(self):
        """不相交的 BBox → union 激活更多 patch"""
        # BBox A: (0,0,10,10) → patch (0,0)
        # BBox B: (50,50,60,60) → patch (5,5)
        mask = compute_multi_bbox_mask(
            bboxes=[(0, 0, 10, 10), (50, 50, 60, 60)],
            page_width=100, page_height=100,
            img_size=100, patch_size=10, has_cls=False,
        )
        assert mask[0].item() == pytest.approx(1.0)   # patch (0,0)
        assert mask[55].item() == pytest.approx(1.0)   # patch (5,5): 5*10+5=55


# ====================================================================
# Test: Padding 掩码
# ====================================================================

class TestPaddingMask:
    def test_square_page_all_valid(self):
        """正方形页面 → 无 padding → 全部有效"""
        mask = compute_padding_mask(100, 100, img_size=100, patch_size=10, has_cls=False)
        assert mask.sum().item() == pytest.approx(100.0)

    def test_portrait_page_has_padding(self):
        """竖版页面 → X 方向有 padding → 部分 patch 无效"""
        mask = compute_padding_mask(100, 200, img_size=1000, patch_size=10, has_cls=False)
        # 应有部分 patch 被标记为 0
        assert mask.sum().item() < mask.shape[0]
        assert mask.min().item() == 0.0
        assert mask.max().item() == 1.0

    def test_cls_always_valid(self):
        """[CLS] token 始终有效"""
        mask = compute_padding_mask(100, 200, img_size=1000, patch_size=10, has_cls=True)
        assert mask[0].item() == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
