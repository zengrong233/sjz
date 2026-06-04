#!/bin/bash
#SBATCH -J lited1r2_refit
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

DATASET="${1:?用法: sbatch $0 rsstod|usod|nwpu}"
JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"

case "$DATASET" in
  rsstod)
    DATA_YAML="data_RS-STOD_PREP_5p13.yaml"
    WEIGHTS="$PROJECT_DIR/best_pt/85036rsstod_best.pt"
    BATCH=6
    LR0=0.002
    PROJECT="runs/rsstod_lited1r2_nwd_refit"
    NAME="rsstod_lited1r2_nwd_b6_e300_from85036_${JOB_ID}"
    ;;
  usod)
    DATA_YAML="data_USOD_PREP_5p13.yaml"
    WEIGHTS="$PROJECT_DIR/best_pt/85210usod_best.pt"
    BATCH=4
    LR0=0.002
    PROJECT="runs/usod_lited1r2_nwd_refit"
    NAME="usod_lited1r2_nwd_b4_e300_from85210_${JOB_ID}"
    ;;
  nwpu)
    DATA_YAML="data_NWPU-VHR10_PREP_5p13.yaml"
    WEIGHTS="$PROJECT_DIR/best_pt/86913nwpu_best.pt"
    BATCH=6
    LR0=0.001
    PROJECT="runs/nwpu_lited1r2_nwd_refit"
    NAME="nwpu_lited1r2_nwd_b6_e300_from86913_${JOB_ID}"
    ;;
  *)
    echo "错误: DATASET 只能是 rsstod/usod/nwpu，当前为: $DATASET"
    exit 1
    ;;
esac

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 未找到模型配置: $MODEL_CFG"
  exit 1
fi

if ! grep -qE '^loss:[[:space:]]*NWD' "$MODEL_CFG"; then
  echo "错误: 当前主线 YAML 不是明确的 loss: NWD"
  grep -nE '^loss:' "$MODEL_CFG" || true
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到初始权重: $WEIGHTS"
  exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
  echo "错误: 未找到数据 YAML: $DATA_YAML"
  exit 1
fi

echo "============================================"
echo "作业ID: $JOB_ID"
echo "数据集: $DATASET"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "数据 YAML: $DATA_YAML"
echo "初始权重: $WEIGHTS"
echo "输出目录: $PROJECT/$NAME"
echo "batch/imgsz/amp/lr0: ${BATCH}/640/false/${LR0}"
echo "loss: NWD"
echo "============================================"

nvidia-smi || true

python -m py_compile \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/modules/block.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python - <<PY
from ultralytics import YOLO
m = YOLO("$MODEL_CFG", task="detect")
m.info(detailed=False)
PY

python "train_yolo111 copy.py" \
  --cfg "$MODEL_CFG" \
  --weights "$WEIGHTS" \
  --data "$DATA_YAML" \
  --device 0 \
  --epochs 300 \
  --patience 80 \
  --imgsz 640 \
  --batch "$BATCH" \
  --workers 8 \
  --project "$PROJECT" \
  --name "$NAME" \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft \
  --small_area_thr 1024 \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_score 0.10 \
  --lr0 "$LR0" \
  --lrf 0.01 \
  --warmup_epochs 0.0 \
  --warmup_bias_lr 0.0 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
