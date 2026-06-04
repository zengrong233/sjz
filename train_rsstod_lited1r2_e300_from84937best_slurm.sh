#!/bin/bash
#SBATCH -J rsstod_e300_84937
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

PROJECT_DIR="/share/home/u2415363072/5.9/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/rsstod_best.pt"

RSSTOD_ROOT="$PROJECT_DIR/datasets/RS-STOD"
if [ ! -d "$RSSTOD_ROOT/images/train" ]; then
  RSSTOD_ROOT="/share/home/u2415363072/5.3/ultralyticsPro--YOLO11/datasets/RS-STOD"
fi
if [ ! -d "$RSSTOD_ROOT/images/train" ]; then
  RSSTOD_ROOT="/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/datasets/RS-STOD"
fi

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 未找到模型配置: $MODEL_CFG"
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到初始权重: $WEIGHTS"
  exit 1
fi

if [ ! -d "$RSSTOD_ROOT/images/train" ] || [ ! -d "$RSSTOD_ROOT/images/val" ]; then
  echo "错误: 未找到 RS-STOD 数据集: $RSSTOD_ROOT"
  exit 1
fi

DATA_YAML="$PROJECT_DIR/data_RS_STOD_5p9_runtime_${JOB_ID}.yaml"
cat > "$DATA_YAML" <<EOF
path: $RSSTOD_ROOT
train: images/train
val: images/val
test: images/test
nc: 5
names: ['Small Vehicle', 'Large Vehicle', 'Ship', 'Airplane', 'Storage Tank']
EOF

rm -f "$RSSTOD_ROOT"/labels/*.cache \
      "$RSSTOD_ROOT"/labels/train.cache \
      "$RSSTOD_ROOT"/labels/val.cache \
      "$RSSTOD_ROOT"/labels/test.cache 2>/dev/null || true

echo "============================================"
echo "作业ID: $JOB_ID"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "初始权重: $WEIGHTS"
echo "数据根目录: $RSSTOD_ROOT"
echo "数据 YAML: $DATA_YAML"
echo "输出目录: runs/rsstod_lited1r2_e300/rsstod_lited1r2_b6_e300_from84937best_${JOB_ID}"
echo "batch/imgsz/amp: 6/640/false"
echo "lr0/lrf/warmup: 0.003/0.01/0"
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
  --batch 6 \
  --workers 8 \
  --project runs/rsstod_lited1r2_e300 \
  --name "rsstod_lited1r2_b6_e300_from84937best_${JOB_ID}" \
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
  --warmup_epochs 0 \
  --warmup_bias_lr 0.003 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
