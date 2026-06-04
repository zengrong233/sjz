#!/bin/bash
#SBATCH -J nwpu_e1nwd100
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
mkdir -p logs runs/nwpu_lited1r2_nwd_e100

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
DATA_YAML="$PROJECT_DIR/data_NWPU-VHR10_5p13.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"

grep -nE '^loss:' "$MODEL_CFG"
grep -qE '^loss:[[:space:]]*NWD' "$MODEL_CFG"

if [ ! -d "$PROJECT_DIR/datasets/NWPU-VHR10/train/images" ]; then
  echo "错误: 未找到 NWPU 数据集"
  exit 1
fi

python -m py_compile \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/modules/block.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python -c "from ultralytics import YOLO; YOLO('$MODEL_CFG', task='detect').info()"

python "train_yolo111 copy.py" \
  --cfg "$MODEL_CFG" \
  --weights "$WEIGHTS" \
  --data "$DATA_YAML" \
  --device 0 \
  --epochs 100 \
  --patience 100 \
  --imgsz 640 \
  --batch 6 \
  --workers 8 \
  --project runs/nwpu_lited1r2_nwd_e100 \
  --name "nwpu_lited1r2_nwd_b6_e100_${JOB_ID}" \
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
