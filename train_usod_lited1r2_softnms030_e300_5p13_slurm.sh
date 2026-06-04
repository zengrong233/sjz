#!/bin/bash
#SBATCH -J usod_s030_e300
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="/share/home/u2415363072/5.13/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"
mkdir -p logs runs/usod_lited1r2_softnms030_train

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"
DATA_YAML="$PROJECT_DIR/data_USOD_PREP_5p13.yaml"

test -f "$MODEL_CFG" || { echo "ERROR: missing model cfg: $MODEL_CFG"; exit 1; }
test -f "$WEIGHTS" || { echo "ERROR: missing weights: $WEIGHTS"; exit 1; }
test -f "$DATA_YAML" || { echo "ERROR: missing data yaml: $DATA_YAML"; exit 1; }

grep -nE '^loss:' "$MODEL_CFG" | grep -q 'NWD' || {
  echo "ERROR: current model is not loss: NWD"
  exit 1
}

python -m py_compile \
  "train_yolo111 copy.py" \
  ultralytics/engine/small_object_trainer.py \
  ultralytics/models/yolo/detect/val.py \
  ultralytics/utils/ops.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

echo "============================================"
echo "JOB_ID=$JOB_ID"
echo "实验: USOD LiteD1R2 + NWD clean train, val/best by Soft-NMS sigma=0.3"
echo "模型: $MODEL_CFG"
echo "数据: $DATA_YAML"
echo "起点: $WEIGHTS"
echo "输出: runs/usod_lited1r2_softnms030_train/usod_softnms030_yolo11n_b4_e300_${JOB_ID}"
echo "注意: Soft-NMS 只影响 val/best selection，不影响训练 loss"
echo "============================================"

python - <<'PY'
from pathlib import Path
from ultralytics.engine.small_object_trainer import SmallObjectABTrainer

project = Path("/share/home/u2415363072/5.13/ultralyticsPro--YOLO11")
job_id = __import__("os").environ.get("SLURM_JOB_ID", "manual")

model_cfg = "YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
weights = str(project / "best_pt/yolo11n.pt")
data_yaml = str(project / "data_USOD_PREP_5p13.yaml")

ab_cfg = {
    "trainer_mode": "full",

    # A: Scale-Routed Optimizer
    "backbone_lr_scale": 0.5,
    "smallobj_lr_scale": 1.25,
    "head_lr_scale": 1.5,
    "smallobj_beta2": 0.9995,
    "smallobj_wd_scale": 0.5,
    "smallobj_grad_clip": 5.0,

    # B: curriculum
    "tiny_area_thr": 256,
    "density_norm_thr": 50.0,
    "alpha": 0.7,
    "beta": 0.3,
    "warmup_curriculum_epochs": 3,
    "base_accum": 0,
    "medium_accum_scale": 1.25,
    "hard_accum_scale": 1.5,
    "max_accum": 24,
    "debug_routing": False,

    # SSDS
    "enable_ssds": True,
    "ssds_mode": "soft",
    "small_area_thr": 1024,
    "tiny_boost": 1.5,
    "small_boost": 1.3,
    "ssds_p3_fallback": True,
    "ssds_p3_fallback_topk": 1,
    "ssds_p3_fallback_score": 0.10,
    "ssds_p3_fallback_min_area": 64.0,
    "ssds_p3_fallback_max_area": 0.0,
}

overrides = {
    "model": model_cfg,
    "pretrained": weights,
    "data": data_yaml,
    "task": "detect",

    "epochs": 300,
    "patience": 100,
    "imgsz": 640,
    "batch": 4,
    "workers": 8,
    "device": 0,
    "seed": 0,
    "amp": False,

    "project": "runs/usod_lited1r2_softnms030_train",
    "name": f"usod_softnms030_yolo11n_b4_e300_{job_id}",
    "exist_ok": False,

    # 与主线 clean 训练一致
    "lr0": 0.01,
    "lrf": 0.01,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "close_mosaic": 10,
    "mosaic": 1.0,
    "scale": 0.5,
    "translate": 0.1,

    # 本实验核心：验证与 best.pt 选择使用 Soft-NMS σ=0.3
    "soft_nms": True,
    "soft_nms_sigma": 0.3,
    "iou": 0.7,
    "conf": 0.001,
}

trainer = SmallObjectABTrainer(overrides=overrides, ab_cfg=ab_cfg)
trainer.train()
PY

echo "训练完成，退出码: $?"
