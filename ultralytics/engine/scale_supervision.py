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
        p3_fallback=False,
        p3_fallback_topk=1,
        p3_fallback_score=0.2,
        p3_fallback_min_area=64.0,
        p3_fallback_max_area=0.0,
    ):
        self.tiny_area_thr = tiny_area_thr
        self.small_area_thr = small_area_thr
        self.tiny_boost = tiny_boost
        self.small_boost = small_boost
        self.mode = mode
        self.enabled = enabled
        self.p3_fallback = p3_fallback
        self.p3_fallback_topk = max(int(p3_fallback_topk), 1)
        self.p3_fallback_score = float(p3_fallback_score)
        self.p3_fallback_min_area = float(p3_fallback_min_area)
        self.p3_fallback_max_area = float(p3_fallback_max_area) if p3_fallback_max_area else float(small_area_thr)

        # 统计日志
        self.stats = {
            "tiny_p2_ratio": 0.0,
            "small_p3_ratio": 0.0,
            "small_p3_native_ratio": 0.0,
            "small_p3_fallback_ratio": 0.0,
            "small_count": 0,
            "fallback_count": 0,
            "calls": 0,
        }

    def reweight(
        self,
        target_scores,
        target_bboxes,
        fg_mask,
        n_anchors_per_level,
        stride_per_anchor,
        anchor_points=None,
        gt_labels=None,
        gt_bboxes=None,
        mask_gt=None,
    ):
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

        target_scores = target_scores * weight

        fallback_count = 0
        fallback_gt_count = 0
        if self.p3_fallback:
            target_scores, target_bboxes, fg_mask, fallback_count, fallback_gt_count = self._apply_p3_fallback(
                target_scores=target_scores,
                target_bboxes=target_bboxes,
                fg_mask=fg_mask,
                p3_mask=p3_mask,
                stride_per_anchor=stride_per_anchor,
                anchor_points=anchor_points,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                mask_gt=mask_gt,
            )

        # 更新统计
        self.stats["calls"] += 1
        if fg_mask.sum() > 0:
            total_tiny = is_tiny.sum().item()
            total_small = is_small.sum().item()
            if total_tiny > 0:
                self.stats["tiny_p2_ratio"] = tiny_on_p2.sum().item() / max(total_tiny, 1)
            if total_small > 0:
                native_ratio = small_on_p3.sum().item() / max(total_small, 1)
                fallback_ratio = fallback_gt_count / max(total_small, 1)
                self.stats["small_p3_native_ratio"] = native_ratio
                self.stats["small_p3_fallback_ratio"] = fallback_ratio
                self.stats["small_p3_ratio"] = min(native_ratio + fallback_ratio, 1.0)
                self.stats["small_count"] = int(total_small)
            elif fallback_gt_count > 0:
                # RS-STOD 等极小目标数据可能没有原生 small 正样本，但 fallback 仍在给 P3 弱监督。
                self.stats["small_p3_native_ratio"] = 0.0
                self.stats["small_p3_fallback_ratio"] = 1.0
                self.stats["small_p3_ratio"] = 1.0
                self.stats["small_count"] = 0
            self.stats["fallback_count"] = int(fallback_count)

        return target_scores, target_bboxes, fg_mask

    def _apply_p3_fallback(
        self,
        target_scores,
        target_bboxes,
        fg_mask,
        p3_mask,
        stride_per_anchor,
        anchor_points,
        gt_labels,
        gt_bboxes,
        mask_gt,
    ):
        """为 small/near-tiny GT 补少量 P3 弱正样本，避免 P3 分支长期无监督。"""
        if anchor_points is None or gt_labels is None or gt_bboxes is None or mask_gt is None:
            return target_scores, target_bboxes, fg_mask, 0, 0

        p3_indices = torch.where(p3_mask)[0]
        if p3_indices.numel() == 0:
            return target_scores, target_bboxes, fg_mask, 0, 0

        stride_vals = stride_per_anchor.view(-1, 1)
        anchor_centers = anchor_points * stride_vals
        p3_centers = anchor_centers[p3_indices]
        fallback_count = 0
        fallback_gt_count = 0

        for b in range(target_scores.shape[0]):
            valid_gt = mask_gt[b, :, 0].bool()
            if not valid_gt.any():
                continue

            boxes = gt_bboxes[b, valid_gt]
            labels = gt_labels[b, valid_gt, 0].long().clamp_(0, target_scores.shape[-1] - 1)
            wh = (boxes[:, 2:4] - boxes[:, 0:2]).clamp(min=0)
            areas_gt = wh[:, 0] * wh[:, 1]
            candidate = (areas_gt >= self.p3_fallback_min_area) & (areas_gt < self.p3_fallback_max_area)
            if not candidate.any():
                continue

            for box, label in zip(boxes[candidate], labels[candidate]):
                center = (box[:2] + box[2:]) * 0.5
                inside = (
                    (p3_centers[:, 0] >= box[0]) &
                    (p3_centers[:, 0] <= box[2]) &
                    (p3_centers[:, 1] >= box[1]) &
                    (p3_centers[:, 1] <= box[3])
                )
                candidate_indices = p3_indices[inside] if inside.any() else p3_indices
                candidate_centers = anchor_centers[candidate_indices]
                dist = ((candidate_centers - center) ** 2).sum(dim=1)
                k = min(self.p3_fallback_topk, candidate_indices.numel())
                selected = candidate_indices[dist.topk(k, largest=False).indices]

                target_bboxes[b, selected] = box
                fg_mask[b, selected] = True
                target_scores[b, selected, label] = torch.maximum(
                    target_scores[b, selected, label],
                    target_scores.new_full((selected.numel(),), self.p3_fallback_score),
                )
                fallback_count += int(selected.numel())
                fallback_gt_count += 1

        return target_scores, target_bboxes, fg_mask, fallback_count, fallback_gt_count

    def get_summary(self):
        """返回统计摘要。"""
        return (
            f"tiny_p2={self.stats['tiny_p2_ratio']:.3f} | "
            f"small_p3={self.stats['small_p3_ratio']:.3f} | "
            f"p3_fb={self.stats['small_p3_fallback_ratio']:.3f}/{self.stats['fallback_count']} | "
            f"calls={self.stats['calls']}"
        )

    def log_stats(self):
        """打印统计信息。"""
        if self.stats["calls"] > 0:
            LOGGER.info(f"{colorstr('SSDS:')} {self.get_summary()}")
