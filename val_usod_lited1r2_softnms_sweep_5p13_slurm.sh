#!/bin/bash
# USOD val-only post-processing ablation on the fixed 85210 E1 anchor.
# No weights or training parameters are changed.

#SBATCH -J usod_softnms_val
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/slurm_usod_softnms_%j.out
#SBATCH -e logs/slurm_usod_softnms_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.13/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"

JOB_ID="${SLURM_JOB_ID:-manual}"
WEIGHTS="${WEIGHTS:-$PROJECT_DIR/best_pt/85210usod_best.pt}"
DATA_YAML="${DATA_YAML:-$PROJECT_DIR/data_USOD_PREP_5p13.yaml}"
PROJECT_OUT="runs/usod_lited1r2_softnms_sweep"
SUMMARY="$PROJECT_DIR/$PROJECT_OUT/softnms_summary_${JOB_ID}.csv"

mkdir -p logs "$PROJECT_OUT"

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到 85210 锚点权重: $WEIGHTS"
  exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
  echo "错误: 未找到 USOD PREP 数据 YAML: $DATA_YAML"
  exit 1
fi

if ! grep -q "soft_nms" ultralytics/models/yolo/detect/val.py; then
  echo "错误: 当前 5.13 代码未检测到 detect validator 的 soft_nms 接口"
  exit 1
fi

if ! grep -q "soft_nms_sigma" ultralytics/cfg/default.yaml; then
  echo "错误: 当前 5.13 配置未注册 soft_nms_sigma"
  exit 1
fi

echo "============================================"
echo "实验: USOD 85210 固定权重的 Soft-NMS val-only sweep"
echo "项目目录: $PROJECT_DIR"
echo "锚点权重: $WEIGHTS"
echo "数据 YAML: $DATA_YAML"
echo "输出目录: $PROJECT_OUT"
echo "对拍组: hard / sigma=0.3(强抑制) / 0.5(默认) / 0.7(弱抑制)"
echo "判据: mAP50-95 >= 0.33649 且 Precision >= 0.84000"
echo "说明: 不训练，不改变 E1/NWD/SSDS/模型结构"
echo "============================================"

nvidia-smi || true

python -m py_compile \
  ultralytics/models/yolo/detect/val.py \
  ultralytics/utils/ops.py \
  ultralytics/cfg/__init__.py

export WEIGHTS DATA_YAML PROJECT_OUT SUMMARY JOB_ID

python - <<'PY'
import csv
import os
from pathlib import Path

from ultralytics import YOLO

weights = Path(os.environ["WEIGHTS"]).resolve()
data_yaml = Path(os.environ["DATA_YAML"]).resolve()
project_out = os.environ["PROJECT_OUT"]
summary = Path(os.environ["SUMMARY"])
job_id = os.environ["JOB_ID"]

anchor_map = 0.33549
map_gate = anchor_map + 0.001
precision_gate = 0.84000

# Gaussian Soft-NMS implementation: exp(-(IoU ** 2) / sigma).
# A smaller sigma therefore applies stronger suppression.
cases = [
    ("v0_hard_nms", False, 0.5, "standard hard NMS"),
    ("v1_soft_sigma03_strong", True, 0.3, "Soft-NMS strong suppression"),
    ("v2_soft_sigma05_default", True, 0.5, "Soft-NMS default suppression"),
    ("v3_soft_sigma07_weak", True, 0.7, "Soft-NMS weak suppression"),
]

rows = []
for tag, use_soft_nms, sigma, note in cases:
    print("=" * 88)
    print(f"[VAL] {tag}: {note}")
    print(f"[POST] soft_nms={use_soft_nms}, sigma={sigma}")

    model = YOLO(str(weights), task="detect")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=4,
        device=0,
        workers=8,
        split="val",
        conf=0.001,
        iou=0.7,
        max_det=300,
        soft_nms=use_soft_nms,
        soft_nms_sigma=sigma,
        project=project_out,
        name=f"{tag}_{job_id}",
        plots=False,
        verbose=True,
    )
    box = metrics.box
    row = {
        "tag": tag,
        "soft_nms": use_soft_nms,
        "sigma": sigma,
        "note": note,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "delta_vs_anchor_map50_95": float(box.map) - anchor_map,
        "passes_gate": float(box.map) >= map_gate and float(box.mp) >= precision_gate,
    }
    rows.append(row)
    print(f"[RESULT] {row}")

summary.parent.mkdir(parents=True, exist_ok=True)
with summary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

rows.sort(key=lambda row: row["map50_95"], reverse=True)
print("=" * 88)
print(f"[SUMMARY] {summary}")
for row in rows:
    print(
        f"{row['tag']}: P={row['precision']:.5f}, R={row['recall']:.5f}, "
        f"mAP50={row['map50']:.5f}, mAP50-95={row['map50_95']:.5f}, "
        f"delta={row['delta_vs_anchor_map50_95']:+.5f}, pass={row['passes_gate']}"
    )
PY

echo "Soft-NMS val-only sweep 完成，汇总文件: $SUMMARY"
