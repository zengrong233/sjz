#!/bin/bash
#SBATCH --job-name=usod_best2_ft
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

rm -f "$PROJECT_DIR/datasets/USOD/labels/train.cache"
rm -f "$PROJECT_DIR/datasets/USOD/labels/val.cache"
rm -f "$PROJECT_DIR/datasets/USOD/labels/test.cache"

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
WEIGHTS="$PROJECT_DIR/runs/usod_p1_nosac_fullc2f_usodbest_finetune/usod_nosac_fullc2f_usodbest_lr_low_b4_e150_seed0_84763/weights/best.pt"

IMGSZ=640
BATCH=4
SEED=0

PROJECT_OUT=runs/usod_p1_nosac_fullc2f_best2_finetune
RUN_NAME=usod_nosac_fullc2f_best2_lr_low_b4_e120_seed${SEED}_${SLURM_JOB_ID}

echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL"
echo "数据 YAML: $DATA"
echo "初始权重: $WEIGHTS"
echo "输出目录: $PROJECT_OUT/$RUN_NAME"
echo "策略: 当前 best.pt 二次低学习率精修，已清 USOD cache"
echo "============================================"

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到初始权重: $WEIGHTS"
  exit 1
fi

python "train_yolo111 copy.py" \
  --cfg "$MODEL" \
  --data "$DATA" \
  --weights "$WEIGHTS" \
  --epochs 120 \
  --patience 60 \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device 0 \
  --workers 8 \
  --project "$PROJECT_OUT" \
  --name "$RUN_NAME" \
  --trainer_mode full \
  --lr0 0.01 \
  --lrf 0.01 \
  --warmup_epochs 0.0 \
  --warmup_momentum 0.937 \
  --warmup_bias_lr 0.0 \
  --backbone_lr_scale 0.04 \
  --smallobj_lr_scale 0.15 \
  --head_lr_scale 0.20 \
  --base_accum 16 \
  --max_accum 24 \
  --enable_ssds \
  --seed "$SEED"

echo "训练完成，退出码: $?"
