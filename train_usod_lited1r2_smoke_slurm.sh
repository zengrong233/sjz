#!/bin/bash
#SBATCH -J usod_lited1r2
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

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.3/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
#WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"
WEIGHTS="$PROJECT_DIR/runs/usod_lited1r2_smoke/usod_lited1r2_b4_e100_85139/weights/best.pt"
USOD_ROOT="${USOD_ROOT:-$PROJECT_DIR/datasets/USOD}"
if [ ! -d "$USOD_ROOT/images/train" ]; then
  USOD_ROOT="/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/datasets/USOD"
fi

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 未找到模型配置: $MODEL_CFG"
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到初始权重: $WEIGHTS"
  exit 1
fi

if [ ! -d "$USOD_ROOT/images/train" ] || [ ! -d "$USOD_ROOT/images/val" ]; then
  echo "错误: 未找到 USOD 数据集: $USOD_ROOT"
  exit 1
fi

DATA_YAML="$PROJECT_DIR/data_USOD_5p3_runtime_${JOB_ID}.yaml"
cat > "$DATA_YAML" <<EOF
path: $USOD_ROOT
train: images/train
val: images/val
test: images/test
nc: 1
names: ['vehicle']
EOF

rm -f "$USOD_ROOT/labels/train.cache" "$USOD_ROOT/labels/val.cache" "$USOD_ROOT/labels/test.cache"

echo "============================================"
echo "作业ID: $JOB_ID"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "数据根目录: $USOD_ROOT"
echo "数据 YAML: $DATA_YAML"
echo "初始权重: $WEIGHTS"
echo "输出目录: runs/usod_lited1r2_smoke/usod_lited1r2_b4_e300_from85139_${JOB_ID}"
echo "batch/imgsz/amp: 4/640/false"
echo "============================================"

nvidia-smi || true

python -m py_compile \
  ultralytics/nn/core11/GDM.py \
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
  --batch 4 \
  --workers 8 \
  --project runs/usod_lited1r2_smoke \
  --name "usod_lited1r2_b4_e300_from85139_${JOB_ID}" \
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