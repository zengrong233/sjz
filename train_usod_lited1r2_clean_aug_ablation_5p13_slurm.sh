#!/bin/bash
# USOD clean augmentation ablation:
#   u0  = yolo11n.pt + historical augmentation settings
#   u1a = yolo11n.pt + mosaic disabled only
#
# Submit examples:
#   sbatch --job-name=usod_u0  train_usod_lited1r2_clean_aug_ablation_5p13_slurm.sh u0
#   sbatch --job-name=usod_u1a train_usod_lited1r2_clean_aug_ablation_5p13_slurm.sh u1a

#SBATCH -J usod_clean_aug
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
mkdir -p logs runs/usod_lited1r2_clean_aug_ablation

VARIANT="${1:-}"
case "$VARIANT" in
  u0)
    MOSAIC=1.0
    EXP_DESC="U0 clean baseline: default augmentation"
    ;;
  u1a)
    MOSAIC=0.0
    EXP_DESC="U1a clean ablation: mosaic disabled only"
    ;;
  *)
    echo "用法: sbatch $0 {u0|u1a}"
    exit 2
    ;;
esac

JOB_ID="${SLURM_JOB_ID:-manual}"
MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"
DATA_YAML="$PROJECT_DIR/data_USOD_PREP_5p13.yaml"
PROJECT_OUT="runs/usod_lited1r2_clean_aug_ablation"
RUN_NAME="usod_${VARIANT}_clean_b4_e300_${JOB_ID}"

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 未找到模型配置: $MODEL_CFG"
  exit 1
fi

if ! grep -Eq '^loss:[[:space:]]*NWD([[:space:]]|$)' "$MODEL_CFG"; then
  echo "错误: 主线 YAML 不是明确的 loss: NWD: $MODEL_CFG"
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到 clean 起点权重: $WEIGHTS"
  exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
  echo "错误: 未找到 USOD PREP 数据 YAML: $DATA_YAML"
  exit 1
fi

USOD_ROOT="$(awk -F ':' '/^[[:space:]]*path[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$DATA_YAML")"

if [ -z "$USOD_ROOT" ] || [ ! -d "$USOD_ROOT/images/train" ] || [ ! -d "$USOD_ROOT/images/val" ]; then
  echo "错误: USOD 数据目录不存在或不完整: ${USOD_ROOT:-<empty>}"
  exit 1
fi

# 不删除 cache。两个首次扫描作业并发写同一 cache 会触发 rename 竞态；
# U0 已结束后，U1a 可以单独扫描，目录不可写时也可不持久化 cache。
if [ ! -f "$USOD_ROOT/labels/train.cache" ] || [ ! -f "$USOD_ROOT/labels/val.cache" ]; then
  echo "提示: cache 不存在，训练进程将自行扫描标签；请确保当前没有另一份首次扫描作业并发运行。"
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "============================================"
echo "作业ID: $JOB_ID"
echo "节点: ${SLURMD_NODENAME:-unknown}"
echo "实验: $EXP_DESC"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG (loss=NWD)"
echo "初始权重: $WEIGHTS (clean from yolo11n.pt)"
echo "数据 YAML: $DATA_YAML"
echo "数据根目录: $USOD_ROOT"
echo "输出目录: $PROJECT_OUT/$RUN_NAME"
echo "增强单变量: mosaic=$MOSAIC scale=0.5 translate=0.1"
echo "训练超参: lr0=0.01 lrf=0.01 warmup=3.0 epochs=300 batch=4 amp=false"
echo "============================================"

nvidia-smi || true

python -m py_compile \
  "train_yolo111 copy.py" \
  ultralytics/nn/core11/GDM.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python -c "from ultralytics import YOLO; m = YOLO('$MODEL_CFG', task='detect'); m.info()"

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
  --seed 0 \
  --project "$PROJECT_OUT" \
  --name "$RUN_NAME" \
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
  --close_mosaic 10 \
  --mosaic "$MOSAIC" \
  --scale 0.5 \
  --translate 0.1

echo "训练完成，退出码: $?"
