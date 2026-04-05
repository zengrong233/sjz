# Ultralytics YOLO 🚀, AGPL-3.0 license
# SmallObjectABTrainer：小目标检测 A+B 联合训练策略
# A: Scale-Routed Optimizer（模块级参数分组 + 差异化学习率）
# B: Noise-Aware Batch Curriculum（基于小目标密度的动态梯度累积）

import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ultralytics.engine.batch_curriculum import BatchCurriculumScheduler
from ultralytics.engine.optimizer_router import (
    build_routed_param_groups,
    print_model_module_tree,
    print_routing_summary,
)
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, colorstr


class SmallObjectABTrainer(DetectionTrainer):
    """
    小目标检测 A+B 联合训练器，继承自 DetectionTrainer。

    支持三种模式：
    - mode='a': 仅启用 Scale-Routed Optimizer
    - mode='b': 仅启用 Noise-Aware Batch Curriculum
    - mode='ab': 同时启用 A+B

    Args:
        cfg: 基础配置
        overrides: 配置覆盖
        _callbacks: 回调
        ab_cfg: A+B 策略专用超参字典
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None, ab_cfg=None):
        # ab_cfg 必须在 super().__init__ 之前保存，因为父类会调用 get_dataset 等
        self.ab_cfg = ab_cfg or {}
        self.trainer_mode = self.ab_cfg.get("trainer_mode", "ab")  # 'a' | 'b' | 'ab' | 'prr' | 'prr_ab' 等
        self.enable_a = self.trainer_mode in ("a", "ab", "prr_ab", "full")
        self.enable_b = self.trainer_mode in ("b", "ab", "prr_ab", "full")
        self.enable_ssds = self.ab_cfg.get("enable_ssds", False) or self.trainer_mode in ("prr_ssds", "full")
        self.debug_routing = self.ab_cfg.get("debug_routing", False)

        # Curriculum 调度器（B 策略）
        self.curriculum = None

        # group-wise grad norm 日志
        self.grad_norm_log = []
        self._initial_group_weight_decays = []
        self._weight_decay_ref_accum = 0

        from ultralytics.utils import DEFAULT_CFG_DICT
        super().__init__(cfg=cfg or DEFAULT_CFG_DICT, overrides=overrides, _callbacks=_callbacks)

    def _resolve_curriculum_base_accum(self, default_accum):
        """优先使用显式 base_accum 覆盖，否则沿用 Ultralytics 自动推导值。"""
        override = int(self.ab_cfg.get("base_accum", 0) or 0)
        return max(override, 1) if override > 0 else max(int(default_accum), 1)

    def _sync_optimizer_weight_decay(self):
        """在动态 accumulate 变化时同步缩放 weight_decay，保持正则强度一致。"""
        optimizer = getattr(self, "optimizer", None)
        ref_accum = max(int(getattr(self, "_weight_decay_ref_accum", 0)), 1)
        if optimizer is None:
            return
        if not self._initial_group_weight_decays:
            self._initial_group_weight_decays = [float(pg.get("weight_decay", 0.0)) for pg in optimizer.param_groups]
        scale = max(int(self.accumulate), 1) / ref_accum
        for pg, base_wd in zip(optimizer.param_groups, self._initial_group_weight_decays):
            pg["weight_decay"] = float(base_wd) * scale

    # ------------------------------------------------------------------
    # A 策略：覆写 build_optimizer
    # ------------------------------------------------------------------
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        构建优化器。当启用 A 策略时使用 Scale-Routed 参数分组，否则回退到父类默认行为。
        """
        if not self.enable_a:
            return super().build_optimizer(model, name, lr, momentum, decay, iterations)

        # 调试：打印模块树
        if self.debug_routing and RANK in {-1, 0}:
            print_model_module_tree(model, max_depth=4)

        # 构建路由参数组
        router_cfg = {
            "backbone_lr_scale": self.ab_cfg.get("backbone_lr_scale", 0.5),
            "smallobj_lr_scale": self.ab_cfg.get("smallobj_lr_scale", 1.25),
            "head_lr_scale": self.ab_cfg.get("head_lr_scale", 1.5),
            "smallobj_beta2": self.ab_cfg.get("smallobj_beta2", 0.9995),
            "smallobj_wd_scale": self.ab_cfg.get("smallobj_wd_scale", 0.5),
        }
        param_groups, stats = build_routed_param_groups(model, lr, decay, cfg=router_cfg)

        # 打印路由摘要
        if RANK in {-1, 0}:
            print_routing_summary(stats, lr, cfg=router_cfg)

        # 统一使用 AdamW（第一版不实现 PRISM 二阶近似）
        # param_groups 中已包含各组的 lr / betas / weight_decay
        optimizer = optim.AdamW(param_groups)

        # 为每组记录 initial_lr（scheduler 需要）
        for pg in optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        LOGGER.info(
            f"{colorstr('optimizer:')} Scale-Routed AdamW with {len(param_groups)} param groups"
        )
        return optimizer

    # ------------------------------------------------------------------
    # B 策略：初始化 curriculum 调度器
    # ------------------------------------------------------------------
    def _setup_train(self, world_size):
        """在父类 _setup_train 之后初始化 curriculum 调度器。"""
        super()._setup_train(world_size)
        self._initial_group_weight_decays = [float(pg.get("weight_decay", 0.0)) for pg in self.optimizer.param_groups]
        self._weight_decay_ref_accum = max(int(self.accumulate), 1)
        self.accumulate = self._resolve_curriculum_base_accum(self.accumulate)
        self._sync_optimizer_weight_decay()

        if self.enable_b:
            self.curriculum = BatchCurriculumScheduler(
                base_accum=self.accumulate,
                tiny_area_thr=self.ab_cfg.get("tiny_area_thr", 256),  # 16*16
                alpha=self.ab_cfg.get("alpha", 0.7),
                beta=self.ab_cfg.get("beta", 0.3),
                ema_momentum=self.ab_cfg.get("ema_momentum", 0.9),
                density_norm_thr=self.ab_cfg.get("density_norm_thr", 50.0),
                warmup_epochs=self.ab_cfg.get("warmup_curriculum_epochs", 3),
                medium_accum_scale=self.ab_cfg.get("medium_accum_scale", 1.25),
                hard_accum_scale=self.ab_cfg.get("hard_accum_scale", 1.5),
                max_accum=self.ab_cfg.get("max_accum", 24),
                enabled=True,
            )
            LOGGER.info(
                f"{colorstr('curriculum:')} Noise-Aware Batch Curriculum 已启用 | "
                f"base_accum={self.accumulate} | tiny_thr={self.curriculum.tiny_area_thr} | "
                f"alpha={self.curriculum.alpha} | beta={self.curriculum.beta} | "
                f"medium_scale={self.curriculum.medium_accum_scale} | "
                f"hard_scale={self.curriculum.hard_accum_scale} | "
                f"max_accum={self.curriculum.max_accum or 'auto'}"
            )

    # ------------------------------------------------------------------
    # SSDS：覆写 init_criterion 使用尺度感知损失
    # ------------------------------------------------------------------
    def init_criterion(self):
        """初始化损失函数。当启用 SSDS 时使用 SSDSDetectionLoss。"""
        if self.enable_ssds:
            from ultralytics.utils.NewLoss.ssds_loss import SSDSDetectionLoss
            from ultralytics.utils.torch_utils import de_parallel
            ssds_cfg = {
                "tiny_area_thr": self.ab_cfg.get("tiny_area_thr", 256),
                "small_area_thr": self.ab_cfg.get("small_area_thr", 1024),
                "tiny_boost": self.ab_cfg.get("tiny_boost", 1.5),
                "small_boost": self.ab_cfg.get("small_boost", 1.3),
                "ssds_mode": self.ab_cfg.get("ssds_mode", "soft"),
                "enable_ssds": True,
            }
            return SSDSDetectionLoss(de_parallel(self.model), ssds_cfg=ssds_cfg)
        return super().init_criterion()


    # ------------------------------------------------------------------
    # B 策略：在 preprocess_batch 中计算 curriculum 指标并更新 accumulate
    # ------------------------------------------------------------------
    def preprocess_batch(self, batch):
        """预处理 batch 并计算 curriculum 难度指标，动态更新 self.accumulate。"""
        batch = super().preprocess_batch(batch)

        if self.enable_b and self.curriculum is not None:
            # 计算难度并更新累积步数
            new_accum = self.curriculum.step(batch, self.epoch, imgsz=self.args.imgsz)
            # DDP 注意：动态 accumulate 在多卡下可能导致各卡不同步
            # 保守策略：DDP 下使用所有卡中的最大 accum 值
            if RANK != -1:
                import torch.distributed as dist
                accum_tensor = torch.tensor([new_accum], dtype=torch.int64, device=self.device)
                dist.all_reduce(accum_tensor, op=dist.ReduceOp.MAX)
                new_accum = int(accum_tensor.item())
            self.accumulate = new_accum
            self._sync_optimizer_weight_decay()

        return batch

    # ------------------------------------------------------------------
    # 覆写 optimizer_step：加入 group-wise grad clipping 和 norm 日志
    # ------------------------------------------------------------------
    def optimizer_step(self):
        """执行优化器步骤，增加 group-wise gradient clipping 和 norm 记录。"""
        self.scaler.unscale_(self.optimizer)

        # group-wise gradient norm 计算和日志
        if self.enable_a and RANK in {-1, 0}:
            norms = {}
            for pg in self.optimizer.param_groups:
                gname = pg.get("group_name", "unknown")
                grads = [p.grad for p in pg["params"] if p.grad is not None]
                if grads:
                    total_norm = torch.norm(
                        torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2
                    ).item()
                    norms[gname] = round(total_norm, 4)
            if norms:
                self.grad_norm_log.append({"epoch": self.epoch, **norms})

        # small_object_group 专属 gradient clipping（可配置）
        smallobj_grad_clip = self.ab_cfg.get("smallobj_grad_clip", 5.0)
        if self.enable_a and smallobj_grad_clip > 0:
            for pg in self.optimizer.param_groups:
                gname = pg.get("group_name", "")
                if "small_object" in gname:
                    torch.nn.utils.clip_grad_norm_(pg["params"], max_norm=smallobj_grad_clip)

        # 全局 gradient clipping（与父类一致）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    # ------------------------------------------------------------------
    # 覆写 save_metrics：追加 curriculum 指标
    # ------------------------------------------------------------------
    def save_metrics(self, metrics):
        """保存指标，追加 curriculum 和 SSDS 相关信息（从第一个 epoch 就写入，保持 CSV 列数一致）。"""
        # B 策略：curriculum 指标
        if self.enable_b:
            if self.curriculum is not None:
                metrics["curriculum/diff_ema"] = round(self.curriculum.diff_ema, 4)
                metrics["curriculum/accum"] = self.curriculum.current_accum
            else:
                metrics["curriculum/diff_ema"] = 0.0
                metrics["curriculum/accum"] = 0
        # SSDS：尺度感知监督指标
        if self.enable_ssds:
            criterion = getattr(self, "criterion", None)
            if criterion is not None and hasattr(criterion, "reweighter"):
                rw = criterion.reweighter
                metrics["ssds/tiny_p2_ratio"] = round(rw.stats.get("tiny_p2_ratio", 0.0), 4)
                metrics["ssds/small_p3_ratio"] = round(rw.stats.get("small_p3_ratio", 0.0), 4)
            else:
                metrics["ssds/tiny_p2_ratio"] = 0.0
                metrics["ssds/small_p3_ratio"] = 0.0
        super().save_metrics(metrics)

    # ------------------------------------------------------------------
    # 覆写 final_eval：训练结束时保存 curriculum 日志
    # ------------------------------------------------------------------
    def final_eval(self):
        """训练结束时保存 curriculum 日志、grad norm 日志和 SSDS 统计。"""
        # 保存 curriculum CSV
        if self.enable_b and self.curriculum is not None:
            self.curriculum.save_log(self.save_dir)

        # 保存 grad norm CSV
        if self.enable_a and self.grad_norm_log:
            import csv
            path = self.save_dir / "grad_norm_log.csv"
            keys = self.grad_norm_log[0].keys()
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.grad_norm_log)
            LOGGER.info(f"Grad norm 日志已保存至 {path}")

        # 打印 SSDS 最终统计
        if self.enable_ssds:
            criterion = getattr(self, "criterion", None)
            if criterion is not None and hasattr(criterion, "reweighter"):
                criterion.reweighter.log_stats()

        return super().final_eval()

    # ------------------------------------------------------------------
    # 覆写 progress_string：显示 curriculum 状态
    # ------------------------------------------------------------------
    def progress_string(self):
        """返回训练进度字符串，附加 curriculum 和 SSDS 状态。"""
        base = super().progress_string()
        if self.enable_b and self.curriculum is not None:
            base += f"\n  [Curriculum] {self.curriculum.get_summary()}"
        if self.enable_ssds:
            criterion = getattr(self, "criterion", None)
            if criterion is not None and hasattr(criterion, "reweighter"):
                base += f"\n  [SSDS] {criterion.reweighter.get_summary()}"
        return base
