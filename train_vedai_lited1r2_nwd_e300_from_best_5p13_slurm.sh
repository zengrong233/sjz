#!/bin/bash
#SBATCH -J vedai_e1nwd300
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
mkdir -p logs runs/vedai_lited1r2_nwd_e300

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/vedai_best.pt"

VEDAI_ROOT="${VEDAI_ROOT:-$PROJECT_DIR/datasets/VEDAI}"
if [ ! -d "$VEDAI_ROOT/images/train" ]; then
  VEDAI_ROOT="/share/home/u2415363072/5.12/ultralyticsPro--YOLO11/datasets/VEDAI"
fi
if [ ! -d "$VEDAI_ROOT/images/train" ]; then
  VEDAI_ROOT="/share/home/u2415363072/5.3/ultralyticsPro--YOLO11/datasets/VEDAI"
fi

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 未找到模型配置: $MODEL_CFG"
  exit 1
fi

if ! grep -qE '^loss:[[:space:]]*NWD' "$MODEL_CFG"; then
  echo "错误: 当前 YAML 不是明确的 loss: NWD"
  grep -nE '^loss:' "$MODEL_CFG" || true
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到热启动权重: $WEIGHTS"
  exit 1
fi

if [ ! -d "$VEDAI_ROOT/images/train" ] || [ ! -d "$VEDAI_ROOT/images/val" ]; then
  echo "错误: 未找到 VEDAI 数据集: $VEDAI_ROOT"
  exit 1
fi

DATA_YAML="$PROJECT_DIR/data_VEDAI_5p13_runtime_e300_${JOB_ID}.yaml"
cat > "$DATA_YAML" <<EOF
path: $VEDAI_ROOT
train: images/train
val: images/val
test: images/test

nc: 7
names:
  0: car
  1: pickup
  2: camping_car
  3: truck
  4: tractor
  5: boat
  6: van
EOF

echo "============================================"
echo "作业ID: $JOB_ID"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "数据根目录: $VEDAI_ROOT"
echo "数据 YAML: $DATA_YAML"
echo "热启动权重: $WEIGHTS"
echo "输出目录: runs/vedai_lited1r2_nwd_e300/vedai_lited1r2_nwd_b6_e300_frombest_${JOB_ID}"
echo "batch/imgsz/amp: 6/640/false"
echo "============================================"

nvidia-smi || true

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
  --epochs 300 \
  --patience 100 \
  --imgsz 640 \
  --batch 6 \
  --workers 8 \
  --project runs/vedai_lited1r2_nwd_e300 \
  --name "vedai_lited1r2_nwd_b6_e300_frombest_${JOB_ID}" \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft \
  --small_area_thr 1024 \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_score 0.10 \
  --lr0 0.003 \
  --lrf 0.01 \
  --warmup_epochs 0.0 \
  --warmup_bias_lr 0.0 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
