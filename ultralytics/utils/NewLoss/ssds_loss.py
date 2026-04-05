# Ultralytics YOLO 🚀, AGPL-3.0 license
# SSDS-enhanced Detection Loss
# 在标准 v8DetectionLoss 基础上加入 Scale-Specific Dual-Branch Supervision

import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.engine.scale_supervision import ScaleSpecificReweighter
from ultralytics.utils import LOGGER, colorstr


class SSDSDetectionLoss(v8DetectionLoss):
    """
    带尺度感知分支监督的检测损失。

    继承 v8DetectionLoss，在 TAL 分配完成后对 target_scores 做尺度感知重加权，
    使 P2 更聚焦 tiny objects，P3 更聚焦 small objects。

    Args:
        model: 检测模型
        tal_topk: TAL top-k 参数
        ssds_cfg: SSDS 配置字典
    """

    def __init__(self, model, tal_topk=10, ssds_cfg=None):
        super().__init__(model, tal_topk)
        cfg = ssds_cfg or {}
        self.reweighter = ScaleSpecificReweighter(
            tiny_area_thr=cfg.get("tiny_area_thr", 256),
            small_area_thr=cfg.get("small_area_thr", 1024),
            tiny_boost=cfg.get("tiny_boost", 1.5),
            small_boost=cfg.get("small_boost", 1.3),
            mode=cfg.get("ssds_mode", "soft"),
            enabled=cfg.get("enable_ssds", True),
        )
        LOGGER.info(
            f"{colorstr('SSDS Loss:')} tiny_thr={self.reweighter.tiny_area_thr} | "
            f"small_thr={self.reweighter.small_area_thr} | "
            f"tiny_boost={self.reweighter.tiny_boost} | "
            f"small_boost={self.reweighter.small_boost} | "
            f"mode={self.reweighter.mode}"
        )

    def __call__(self, preds, batch):
        """计算带 SSDS 重加权的检测损失。"""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        from ultralytics.utils.tal import make_anchors
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # TAL 分配
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # ===== SSDS 重加权 =====
        target_scores = self.reweighter.reweight(
            target_scores, target_bboxes, fg_mask,
            n_anchors_per_level=[f.shape[2] * f.shape[3] for f in feats],
            stride_per_anchor=stride_tensor,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss（使用标准 BCE）
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()
