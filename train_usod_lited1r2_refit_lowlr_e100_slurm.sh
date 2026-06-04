#!/bin/bash
#SBATCH -J usod_lowlr100
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

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.13/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"
mkdir -p logs runs

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/85210usod_best.pt"
DATA_YAML="$PROJECT_DIR/data_USOD_PREP_5p13.yaml"
DATA_ROOT="$PROJECT_DIR/datasets/USOD"

test -f "$MODEL_CFG"
test -f "$WEIGHTS"
test -f "$DATA_YAML"

grep -nE '^loss:[[:space:]]*NWD' "$MODEL_CFG"

rm -f "$DATA_ROOT/labels/train.cache" "$DATA_ROOT/labels/val.cache" "$DATA_ROOT/labels/test.cache" || true

python -m py_compile \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/modules/block.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python "train_yolo111 copy.py" \
  --cfg "$MODEL_CFG" \
  --weights "$WEIGHTS" \
  --data "$DATA_YAML" \
  --device 0 \
  --epochs 100 \
  --patience 80 \
  --imgsz 640 \
  --batch 4 \
  --workers 8 \
  --project runs/usod_lited1r2_refit_lowlr \
  --name "usod_refit_lowlr_from85210_b4_e100_${JOB_ID}" \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft \
  --small_area_thr 1024 \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_score 0.10 \
  --lr0 0.0005 \
  --lrf 0.1 \
  --warmup_epochs 1.0 \
  --warmup_bias_lr 0.0 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
