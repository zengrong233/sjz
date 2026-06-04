#!/bin/bash
#SBATCH --job-name=usod_nosac_b4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/logs/slurm_%j.out
#SBATCH --error=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11
cd "$PROJECT_DIR"

mkdir -p logs runs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cat > data_USOD_4p25_runtime.yaml <<'YAML'
path: /share/home/u2415363072/4.25/ultralyticsPro--YOLO11/datasets/USOD
train: images/train
val: images/val
test:

names:
  0: target
YAML

MODEL=YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml
DATA=data_USOD_4p25_runtime.yaml
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"

IMGSZ=640
BATCH=4
SEED=0

PROJECT_OUT=runs/usod_p1_nosac_fullc2f_640_b4
RUN_NAME=usod_nosac_fullc2f_640_b4_e300_seed${SEED}_${SLURM_JOB_ID}

echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL"
echo "数据 YAML: $DATA"
echo "初始权重: $WEIGHTS"
echo "imgsz/batch/seed/amp: $IMGSZ/$BATCH/$SEED/false"
echo "输出目录: $PROJECT_OUT/$RUN_NAME"
echo "============================================"

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到初始权重: $WEIGHTS"
  exit 1
fi

if [ ! -d "$PROJECT_DIR/datasets/USOD/images/train" ] || [ ! -d "$PROJECT_DIR/datasets/USOD/images/val" ]; then
  echo "错误: USOD 数据集路径不存在或不完整: $PROJECT_DIR/datasets/USOD"
  exit 1
fi

python "train_yolo111 copy.py" \
  --cfg "$MODEL" \
  --data "$DATA" \
  --weights "$WEIGHTS" \
  --epochs 300 \
  --patience 100 \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device 0 \
  --workers 8 \
  --project "$PROJECT_OUT" \
  --name "$RUN_NAME" \
  --trainer_mode full \
  --lr0 0.01 \
  --lrf 0.01 \
  --warmup_epochs 3.0 \
  --warmup_momentum 0.8 \
  --warmup_bias_lr 0.1 \
  --base_accum 16 \
  --max_accum 24 \
  --enable_ssds \
  --seed "$SEED"

echo "训练完成，退出码: $?"
