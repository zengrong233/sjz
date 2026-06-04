#!/bin/bash
#SBATCH -J nwpu_clean300
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
mkdir -p logs runs best_pt

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"
DATA_YAML="$PROJECT_DIR/data_NWPU-VHR10_PREP_5p13.yaml"
DATA_ROOT="$PROJECT_DIR/datasets/NWPU-VHR10"

if [ ! -f "$WEIGHTS" ]; then
  cp -a /share/home/u2415363072/5.3/ultralyticsPro--YOLO11/best_pt/yolo11n.pt "$WEIGHTS"
fi

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
  --epochs 300 \
  --patience 100 \
  --imgsz 640 \
  --batch 6 \
  --workers 8 \
  --project runs/nwpu_lited1r2_clean \
  --name "nwpu_clean_yolo11n_b6_e300_${JOB_ID}" \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft \
  --small_area_thr 1024 \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_score 0.10 \
  --lr0 0.01 \
  --lrf 0.01 \
  --warmup_epochs 3.0 \
  --warmup_bias_lr 0.1 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
