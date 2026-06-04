#!/bin/bash
#SBATCH -J val_nwpu133
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="/share/home/u2415363072/5.13/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"
mkdir -p logs runs/val_nwpu_prep133

WEIGHTS="$PROJECT_DIR/runs/nwpu_lited1r2_clean/nwpu_clean_yolo11n_b6_e300_87416/weights/best.pt"
DATA="$PROJECT_DIR/data_NWPU-VHR10_PREP_5p13.yaml"

test -f "$WEIGHTS" || { echo "错误: 缺少 87416 best.pt: $WEIGHTS"; exit 1; }
test -f "$DATA" || { echo "错误: 缺少数据 YAML: $DATA"; exit 1; }

python - <<'PY'
from ultralytics import YOLO

weights = "/share/home/u2415363072/5.13/ultralyticsPro--YOLO11/runs/nwpu_lited1r2_clean/nwpu_clean_yolo11n_b6_e300_87416/weights/best.pt"
data = "/share/home/u2415363072/5.13/ultralyticsPro--YOLO11/data_NWPU-VHR10_PREP_5p13.yaml"

model = YOLO(weights, task="detect")
metrics = model.val(
    data=data,
    imgsz=640,
    batch=6,
    device=0,
    workers=8,
    project="runs/val_nwpu_prep133",
    name="val_87416_on_prep133",
    split="val",
    plots=True,
    verbose=True,
)

box = metrics.box
print(
    f"[BASELINE_133] P={box.mp:.5f}, R={box.mr:.5f}, "
    f"mAP50={box.map50:.5f}, mAP50-95={box.map:.5f}"
)
PY
