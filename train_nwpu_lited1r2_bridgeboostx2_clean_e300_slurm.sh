#!/bin/bash
#SBATCH -J nwpu_bbx2_clean
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

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"
DATA_YAML="$PROJECT_DIR/data_NWPU-VHR10_PREP_BridgeBoostX2_5p13.yaml"
DATA_ROOT="$PROJECT_DIR/datasets/NWPU-VHR10-PREP-BridgeBoost-x2"

test -f "$MODEL_CFG"
test -f "$WEIGHTS"
test -f "$DATA_YAML"
test -d "$DATA_ROOT/images/train"
test -d "$DATA_ROOT/labels/train"
test -d "$DATA_ROOT/images/val"
test -d "$DATA_ROOT/labels/val"

if ! grep -qE '^loss:[[:space:]]*NWD' "$MODEL_CFG"; then
  echo "错误: 当前主 YAML 不是明确的 loss: NWD"
  exit 1
fi

rm -f \
  "$DATA_ROOT/labels/train.cache" \
  "$DATA_ROOT/labels/val.cache" \
  "$DATA_ROOT/labels/test.cache" \
  "$DATA_ROOT/labels/train.cache.npy" \
  "$DATA_ROOT/labels/val.cache.npy" \
  "$DATA_ROOT/labels/test.cache.npy"

echo "============================================"
echo "作业ID: $JOB_ID"
echo "模型配置: $MODEL_CFG"
echo "初始权重: $WEIGHTS"
echo "数据配置: $DATA_YAML"
echo "数据目录: $DATA_ROOT"
echo "实验变量: bridge train oversampling x2 only"
echo "对照基线: nwpu_clean_yolo11n_b6_e300_87416"
echo "batch/imgsz/epochs: 6/640/300"
echo "============================================"

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
  --project runs/nwpu_lited1r2_bridgeboost_x2_clean \
  --name "nwpu_bridgeboostx2_yolo11n_b6_e300_${JOB_ID}" \
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
