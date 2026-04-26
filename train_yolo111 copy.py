import sys
import argparse
import os
import torch
import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")


def main(opt):
    alias_map = {
        "prr": "baseline",
        "baseline_prr": "baseline",
        "prr_ab": "ab",
        "prr_ssds": "full",
    }
    if opt.trainer_mode in alias_map:
        normalized_mode = alias_map[opt.trainer_mode]
        print(f"提示: trainer_mode={opt.trainer_mode} 已归一化为 {normalized_mode}")
        opt.trainer_mode = normalized_mode

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练")
        device = 'cpu'
    else:
        device = opt.device
        print(f"使用设备: {device}")
        print(f"可用的GPU数量: {torch.cuda.device_count()}")

    yaml = opt.cfg
    weights = opt.weights
    data = opt.data

    # 初始化模型
    model = YOLO(yaml, task='detect')  # 明确指定任务类型
    if weights and os.path.exists(weights):
        model = model.load(weights)

    model.info()

    # ------------------------------------------------------------------
    # 训练模式分发
    # ------------------------------------------------------------------
    if opt.trainer_mode == "baseline":
        # 原始行为，不做任何修改
        train_kwargs = {
            "data": data,
            "epochs": opt.epochs,
            "imgsz": opt.imgsz,
            "workers": opt.workers,
            "batch": opt.batch,
            "device": device,
            "exist_ok": opt.exist_ok,
            "amp": opt.amp,
        }
        if opt.project:
            train_kwargs["project"] = opt.project
        if opt.name:
            train_kwargs["name"] = opt.name
        results = model.train(**train_kwargs)
    else:
        # A / B / AB 模式：使用 SmallObjectABTrainer
        from ultralytics.engine.small_object_trainer import SmallObjectABTrainer

        # 构建 ab_cfg 超参字典
        ab_cfg = {
            "trainer_mode": opt.trainer_mode,
            "backbone_lr_scale": opt.backbone_lr_scale,
            "smallobj_lr_scale": opt.smallobj_lr_scale,
            "head_lr_scale": opt.head_lr_scale,
            "smallobj_beta2": opt.smallobj_beta2,
            "smallobj_wd_scale": opt.smallobj_wd_scale,
            "smallobj_grad_clip": opt.smallobj_grad_clip,
            "tiny_area_thr": opt.tiny_area_thr,
            "density_norm_thr": opt.density_norm_thr,
            "alpha": opt.alpha,
            "beta": opt.beta,
            "warmup_curriculum_epochs": opt.warmup_curriculum_epochs,
            "base_accum": opt.base_accum,
            "medium_accum_scale": opt.medium_accum_scale,
            "hard_accum_scale": opt.hard_accum_scale,
            "max_accum": opt.max_accum,
            "debug_routing": opt.debug_routing,
            # SSDS 参数
            "enable_ssds": opt.enable_ssds,
            "ssds_mode": opt.ssds_mode,
            "small_area_thr": opt.small_area_thr,
            "tiny_boost": opt.tiny_boost,
            "small_boost": opt.small_boost,
        }

        # 构建 overrides（与 model.train() 内部逻辑一致）
        overrides = {
            "model": yaml,
            "data": data,
            "epochs": opt.epochs,
            "imgsz": opt.imgsz,
            "workers": opt.workers,
            "batch": opt.batch,
            "device": device,
            "exist_ok": opt.exist_ok,
            "amp": opt.amp,
            "task": "detect",
        }
        if opt.project:
            overrides["project"] = opt.project
        if opt.name:
            overrides["name"] = opt.name

        # 如果有预训练权重，设置 pretrained
        if weights and os.path.exists(weights):
            overrides["pretrained"] = weights

        trainer = SmallObjectABTrainer(overrides=overrides, ab_cfg=ab_cfg)
        trainer.train()


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description="VisDrone 小目标训练入口，默认使用 PRR-v3 + AsDDet + A+B + SSDS 推荐链路。"
    )
    # 原有参数
    parser.add_argument('--cfg', type=str,
                        default=r'YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml',
                        help='模型 yaml 路径')
    parser.add_argument('--weights', type=str,
                        default=r'yolo11n.pt',
                        help='预训练权重路径')
    parser.add_argument('--data', type=str,
                        default=r'data_VD_slurm.yaml',
                        help='数据集 yaml 路径')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda 设备 (例如 0 或 0,1,2,3 或 cpu)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=8,
                        help='batch size')
    parser.add_argument('--workers', type=int, default=8,
                        help='dataloader workers')
    parser.add_argument('--project', type=str, default='',
                        help='训练输出根目录（可选）')
    parser.add_argument('--name', type=str, default='',
                        help='本次训练名称（可选）')
    parser.add_argument('--exist_ok', action='store_true',
                        help='允许复用已有输出目录（默认关闭，避免 results.csv 追加污染）')
    parser.add_argument('--amp', action='store_true',
                        help='启用混合精度训练，降低显存占用')

    # A+B 策略参数
    parser.add_argument('--trainer_mode', type=str, default='full',
                        choices=['baseline', 'baseline_prr', 'a', 'b', 'ab', 'prr', 'prr_ab', 'prr_ssds', 'full'],
                        help='训练模式: baseline=PRR-v3结构基线, ab=PRR-v3+A+B, full=PRR-v3+A+B+SSDS')
    parser.add_argument('--base_accum', type=int, default=0,
                        help='B策略: 基础梯度累积步数，0 表示自动沿用 nbs/batch 推导值')
    parser.add_argument('--tiny_area_thr', type=int, default=256,
                        help='B策略: 小目标面积阈值 (像素², 默认 16*16=256)')
    parser.add_argument('--density_norm_thr', type=float, default=50.0,
                        help='B策略: 密度归一化阈值')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='B策略: tiny_ratio 权重')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='B策略: density_ratio 权重')
    parser.add_argument('--warmup_curriculum_epochs', type=int, default=3,
                        help='B策略: 禁用动态累积的 warmup epoch 数')
    parser.add_argument('--medium_accum_scale', type=float, default=1.25,
                        help='B策略: 中等难度 batch 的累积倍率')
    parser.add_argument('--hard_accum_scale', type=float, default=1.5,
                        help='B策略: 高难度 batch 的累积倍率')
    parser.add_argument('--max_accum', type=int, default=24,
                        help='B策略: 动态累积步数上限，0 表示不设上限')

    # A 策略参数
    parser.add_argument('--backbone_lr_scale', type=float, default=0.5,
                        help='A策略: backbone 学习率缩放')
    parser.add_argument('--smallobj_lr_scale', type=float, default=1.25,
                        help='A策略: 小目标模块学习率缩放')
    parser.add_argument('--head_lr_scale', type=float, default=1.5,
                        help='A策略: 检测头学习率缩放')
    parser.add_argument('--smallobj_beta2', type=float, default=0.9995,
                        help='A策略: 小目标模块 AdamW beta2')
    parser.add_argument('--smallobj_wd_scale', type=float, default=0.5,
                        help='A策略: 小目标模块 weight_decay 缩放')
    parser.add_argument('--smallobj_grad_clip', type=float, default=5.0,
                        help='A策略: 小目标模块梯度裁剪阈值')
    parser.add_argument('--debug_routing', action='store_true',
                        help='A策略: 打印完整模块路由树')

    # SSDS 监督补强参数
    parser.add_argument('--enable_ssds', action='store_true',
                        help='SSDS: 启用尺度感知分支监督')
    parser.add_argument('--ssds_mode', type=str, default='soft',
                        choices=['soft', 'hard'],
                        help='SSDS: soft=乘法加权, hard=强制分配')
    parser.add_argument('--small_area_thr', type=int, default=1024,
                        help='SSDS: small 目标面积阈值 (像素², 默认 32*32=1024)')
    parser.add_argument('--tiny_boost', type=float, default=1.5,
                        help='SSDS: tiny GT 在 P2 层的权重放大系数')
    parser.add_argument('--small_boost', type=float, default=1.3,
                        help='SSDS: small GT 在 P3 层的权重放大系数')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    # 设置环境变量
    os.environ['ULTRALYTICS_SKIP_DOWNLOAD'] = '1'  # 禁用自动下载

    opt = parse_opt()
    main(opt)
