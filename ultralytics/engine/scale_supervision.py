# Ultralytics YOLO 🚀, AGPL-3.0 license
# Scale-Specific Dual-Branch Supervision（SSDS）
# 监督补强：显式约束 P2 聚焦 tiny objects，P3 聚焦 small objects
# 通过对 TAL 分配结果做尺度感知 soft reweighting 实现

import torch
from ultralytics.utils import LOGGER, colorstr


class ScaleSpecificReweighter:
    """
    尺度感知分支监督重加权器。

    在 TAL 分配完成后，根据 GT box 面积对不同尺度层的 target_scores 做 soft reweighting：
    - tiny GT（面积 < tiny_thr）在 P2 层 anchor 上的权重被放大
    - small GT（tiny_thr <= 面积 < small_thr）在 P3 层 anchor 上的权重被放大
    - medium/large GT 保持不变

    这种方式不改 TAL 分配逻辑本身，只在分配结果上做后处理，风险最低。

    Args:
        tiny_area_thr: tiny 目标面积阈值（像素²，输入尺度下），默认 16*16=256
        small_area_thr: small 目标面积阈值（像素²），默认 32*32=1024
        tiny_boost: tiny GT 在 P2 层的权重放大系数
        small_boost: small GT 在 P3 层的权重放大系数
        mode: 'soft'（默认，乘法加权）或 'hard'（强制只分配到对应层）
        enabled: 是否启用
    """

    def __init__(
        self,
        tiny_area_thr=256,
        small_area_thr=1024,
        tiny_boost=1.5,
        small_boost=1.3,
        mode="soft",
        enabled=True,
    ):
        self.tiny_area_thr = tiny_area_thr
        self.small_area_thr = small_area_thr
        self.tiny_boost = tiny_boost
        self.small_boost = small_boost
        self.mode = mode
        self.enabled = enabled

        # 统计日志
        self.stats = {"tiny_p2_ratio": 0.0, "small_p3_ratio": 0.0, "calls": 0}

    def reweight(self, target_scores, target_bboxes, fg_mask, n_anchors_per_level, stride_per_anchor):
        """
        对 TAL 分配结果做尺度感知重加权。

        Args:
            target_scores: (B, total_anchors, nc) TAL 分配的目标分数
            target_bboxes: (B, total_anchors, 4) TAL 分配的目标框（xyxy，已除以 stride）
            fg_mask: (B, total_anchors) 前景 mask
            n_anchors_per_level: list[int]，每个检测层的 anchor 数量
            stride_per_anchor: (total_anchors,) 或 (total_anchors, 1)，每个 anchor 的 stride

        Returns:
            reweighted target_scores
        """
        if not self.enabled:
            return target_scores

        B, N, nc = target_scores.shape
        fg_mask = fg_mask.bool()

        # target_bboxes 来自 TAL assigner，已经是输入图像尺度下的 xyxy 像素坐标。
        # 这里不能再乘 stride，否则 tiny/small 面积会被错误放大，SSDS 将退化为全不命中。
        w = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp(min=0)
        h = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp(min=0)
        areas = w * h  # (B, N)

        # 确定每个 anchor 属于哪个检测层
        stride_vals = stride_per_anchor.view(-1)
        min_stride = stride_vals.min()
        p2_mask = stride_vals == min_stride  # 最小 stride 视作 P2
        unique_strides = stride_vals.unique().sort()[0]
        if len(unique_strides) >= 2:
            p3_mask = stride_vals == unique_strides[1]
        else:
            p3_mask = torch.zeros_like(p2_mask)

        # 构建权重矩阵
        weight = torch.ones(B, N, 1, device=target_scores.device, dtype=target_scores.dtype)

        # tiny GT 在 P2 层加权
        is_tiny = (areas < self.tiny_area_thr) & fg_mask  # (B, N)
        tiny_on_p2 = is_tiny & p2_mask.unsqueeze(0)  # (B, N)

        # small GT 在 P3 层加权
        is_small = (areas >= self.tiny_area_thr) & (areas < self.small_area_thr) & fg_mask
        small_on_p3 = is_small & p3_mask.unsqueeze(0)

        if self.mode == "soft":
            weight = torch.where(tiny_on_p2.unsqueeze(-1), weight.new_full((1,), self.tiny_boost), weight)
            weight = torch.where(small_on_p3.unsqueeze(-1), weight.new_full((1,), self.small_boost), weight)
        elif self.mode == "hard":
            # hard 模式：tiny GT 不在 P2 上的权重降为 0.1
            tiny_not_p2 = is_tiny & (~p2_mask.unsqueeze(0))
            weight = torch.where(tiny_not_p2.unsqueeze(-1), weight.new_full((1,), 0.1), weight)
            weight = torch.where(tiny_on_p2.unsqueeze(-1), weight.new_full((1,), self.tiny_boost), weight)
            small_not_p3 = is_small & (~p3_mask.unsqueeze(0))
            weight = torch.where(small_not_p3.unsqueeze(-1), weight.new_full((1,), 0.1), weight)
            weight = torch.where(small_on_p3.unsqueeze(-1), weight.new_full((1,), self.small_boost), weight)

        # 更新统计
        self.stats["calls"] += 1
        if fg_mask.sum() > 0:
            total_tiny = is_tiny.sum().item()
            total_small = is_small.sum().item()
            if total_tiny > 0:
                self.stats["tiny_p2_ratio"] = tiny_on_p2.sum().item() / max(total_tiny, 1)
            if total_small > 0:
                self.stats["small_p3_ratio"] = small_on_p3.sum().item() / max(total_small, 1)

        return target_scores * weight

    def get_summary(self):
        """返回统计摘要。"""
        return (
            f"tiny_p2={self.stats['tiny_p2_ratio']:.3f} | "
            f"small_p3={self.stats['small_p3_ratio']:.3f} | "
            f"calls={self.stats['calls']}"
        )

    def log_stats(self):
        """打印统计信息。"""
        if self.stats["calls"] > 0:
            LOGGER.info(f"{colorstr('SSDS:')} {self.get_summary()}")
