#!/bin/bash
#SBATCH -J rsstod_ssalite300
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.12/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"
mkdir -p logs runs

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2-SSALite.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"

RSSTOD_ROOT="${RSSTOD_ROOT:-$PROJECT_DIR/datasets/RS-STOD}"
if [ ! -d "$RSSTOD_ROOT/images/train" ]; then
  RSSTOD_ROOT="/share/home/u2415363072/5.3/ultralyticsPro--YOLO11/datasets/RS-STOD"
fi
if [ ! -d "$RSSTOD_ROOT/images/train" ]; then
  RSSTOD_ROOT="/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/datasets/RS-STOD"
fi

test -f "$MODEL_CFG" || { echo "缺少模型配置: $MODEL_CFG"; exit 1; }
test -f "$WEIGHTS" || { echo "缺少初始权重: $WEIGHTS"; exit 1; }
test -d "$RSSTOD_ROOT/images/train" || { echo "缺少 RS-STOD 数据: $RSSTOD_ROOT"; exit 1; }
test -d "$RSSTOD_ROOT/images/val" || { echo "缺少 RS-STOD val: $RSSTOD_ROOT"; exit 1; }

DATA_YAML="$PROJECT_DIR/data_RS_STOD_5p12_ssalite_e300_${JOB_ID}.yaml"
cat > "$DATA_YAML" <<EOF
path: $RSSTOD_ROOT
train: images/train
val: images/val
test: images/test
nc: 5
names: ['Small Vehicle', 'Large Vehicle', 'Ship', 'Airplane', 'Storage Tank']
EOF

echo "============================================"
echo "JOB_ID: $JOB_ID"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "MODEL_CFG: $MODEL_CFG"
echo "WEIGHTS: $WEIGHTS"
echo "RSSTOD_ROOT: $RSSTOD_ROOT"
echo "DATA_YAML: $DATA_YAML"
echo "OUTPUT: runs/rsstod_lited1r2_ssalite_e300/rsstod_lited1r2_ssalite_b6_e300_${JOB_ID}"
echo "batch/imgsz/amp: 6/640/false"
echo "============================================"

nvidia-smi || true

python -m py_compile \
  ultralytics/nn/modules/block.py \
  ultralytics/nn/modules/__init__.py \
  ultralytics/nn/tasks.py \
  ultralytics/nn/core11/GDM.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python -c "from ultralytics import YOLO; YOLO('$MODEL_CFG', task='detect').info(detailed=False)"

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
  --project runs/rsstod_lited1r2_ssalite_e300 \
  --name "rsstod_lited1r2_ssalite_b6_e300_${JOB_ID}" \
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
