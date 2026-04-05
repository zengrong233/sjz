# Ultralytics YOLO 🚀, AGPL-3.0 license
# Noise-Aware Batch Curriculum：基于小目标密度的动态梯度累积调度
# 根据每个 batch 的小目标比例和密度动态调整 gradient accumulation 步数

import csv
import math
import os
import torch
from ultralytics.utils import LOGGER, colorstr


class BatchCurriculumScheduler:
    """
    Noise-Aware Batch Curriculum 调度器。

    根据每个 batch 中小目标的比例和密度，动态调整梯度累积步数。
    难度越高的 batch 使用更大的有效 batch size（更多累积步），以降低梯度噪声。

    Args:
        base_accum: 基础累积步数
        tiny_area_thr: 小目标面积阈值（像素²，按输入尺度）
        alpha: tiny_ratio 权重
        beta: density_ratio 权重
        ema_momentum: 难度 EMA 平滑系数
        density_norm_thr: 密度归一化阈值（每张图平均目标数上限）
        warmup_epochs: 禁用动态累积的 warmup epoch 数
        enabled: 是否启用
    """

    def __init__(
        self,
        base_accum=1,
        tiny_area_thr=256,  # 16*16
        alpha=0.7,
        beta=0.3,
        ema_momentum=0.9,
        density_norm_thr=50.0,
        warmup_epochs=3,
        medium_accum_scale=1.25,
        hard_accum_scale=1.5,
        max_accum=24,
        enabled=True,
    ):
        self.base_accum = base_accum
        self.tiny_area_thr = tiny_area_thr
        self.alpha = alpha
        self.beta = beta
        self.ema_momentum = ema_momentum
        self.density_norm_thr = density_norm_thr
        self.warmup_epochs = warmup_epochs
        self.medium_accum_scale = max(float(medium_accum_scale), 1.0)
        self.hard_accum_scale = max(float(hard_accum_scale), self.medium_accum_scale)
        self.max_accum = max(int(max_accum), 0)
        self.enabled = enabled

        # 运行时状态
        self.diff_ema = 0.0
        self.current_accum = base_accum
        self.step_count = 0

        # 日志记录
        self.history = []

    def _scaled_accum(self, scale):
        """按给定倍率计算累积步数，并限制最小/最大边界。"""
        accum = max(int(math.ceil(self.base_accum * scale)), self.base_accum)
        if self.max_accum > 0:
            accum = min(accum, self.max_accum)
        return accum

    def compute_batch_difficulty(self, batch, imgsz=640):
        """
        从 batch 的 targets 中计算难度指标。

        Args:
            batch: 训练 batch 字典，包含 'bboxes'（归一化 xywh）、'batch_idx'、'img'
            imgsz: 输入图像尺寸

        Returns:
            dict: 包含 tiny_ratio, density_ratio, difficulty
        """
        bboxes = batch.get("bboxes", None)  # (N, 4) 归一化 xywh
        batch_idx = batch.get("batch_idx", None)  # (N,)

        if bboxes is None or len(bboxes) == 0:
            return {"tiny_ratio": 0.0, "density_ratio": 0.0, "difficulty": 0.0}

        # 计算面积（归一化 -> 像素²）
        # bboxes 格式: xywh 归一化，w 和 h 在 index 2, 3
        w_px = bboxes[:, 2] * imgsz
        h_px = bboxes[:, 3] * imgsz
        areas = w_px * h_px

        # tiny_ratio: 小目标占比
        n_total = len(bboxes)
        n_tiny = (areas < self.tiny_area_thr).sum().item()
        tiny_ratio = n_tiny / max(n_total, 1)

        # density_ratio: 每张图平均目标数，归一化到 [0, 1]
        if batch_idx is not None:
            n_images = int(batch_idx.max().item()) + 1 if len(batch_idx) > 0 else 1
        else:
            n_images = batch["img"].shape[0] if "img" in batch else 1
        avg_targets = n_total / max(n_images, 1)
        density_ratio = min(avg_targets / self.density_norm_thr, 1.0)

        # 综合难度
        difficulty = self.alpha * tiny_ratio + self.beta * density_ratio

        return {"tiny_ratio": tiny_ratio, "density_ratio": density_ratio, "difficulty": difficulty}

    def step(self, batch, epoch, imgsz=640):
        """
        根据当前 batch 更新累积步数。

        Args:
            batch: 训练 batch
            epoch: 当前 epoch
            imgsz: 输入图像尺寸

        Returns:
            int: 当前应使用的累积步数
        """
        if not self.enabled:
            return self.base_accum

        metrics = self.compute_batch_difficulty(batch, imgsz)
        difficulty = metrics["difficulty"]

        # EMA 平滑
        self.diff_ema = self.ema_momentum * self.diff_ema + (1 - self.ema_momentum) * difficulty

        # warmup 期间使用固定累积
        if epoch < self.warmup_epochs:
            self.current_accum = self.base_accum
        else:
            # 根据 diff_ema 选择累积倍数。默认使用温和倍率，避免过早将累积步数顶到 2x/3x。
            if self.diff_ema < 0.3:
                self.current_accum = self.base_accum  # easy
            elif self.diff_ema < 0.6:
                self.current_accum = self._scaled_accum(self.medium_accum_scale)  # medium
            else:
                self.current_accum = self._scaled_accum(self.hard_accum_scale)  # hard

        # 记录
        self.step_count += 1
        record = {
            "step": self.step_count,
            "epoch": epoch,
            "tiny_ratio": round(metrics["tiny_ratio"], 4),
            "density_ratio": round(metrics["density_ratio"], 4),
            "difficulty": round(metrics["difficulty"], 4),
            "diff_ema": round(self.diff_ema, 4),
            "accum": self.current_accum,
        }
        self.history.append(record)

        return self.current_accum

    def save_log(self, save_dir):
        """将 curriculum 日志写入 CSV 文件。"""
        if not self.history:
            return
        path = os.path.join(str(save_dir), "curriculum_log.csv")
        keys = self.history[0].keys()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
        LOGGER.info(f"Curriculum 日志已保存至 {path}")

    def get_summary(self):
        """返回当前状态摘要字符串。"""
        return (
            f"diff_ema={self.diff_ema:.4f} | accum={self.current_accum} | "
            f"steps={self.step_count} | enabled={self.enabled}"
        )
