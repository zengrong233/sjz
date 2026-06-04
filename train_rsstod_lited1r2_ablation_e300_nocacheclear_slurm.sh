#!/bin/bash
#SBATCH -J rsstod_ablate300
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

MODEL_CFG="${1:?need MODEL_CFG}"
TAG="${2:?need TAG}"

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.13/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"
mkdir -p logs runs

JOB_ID="${SLURM_JOB_ID:-manual}"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"

RSSTOD_ROOT="${RSSTOD_ROOT:-$PROJECT_DIR/datasets/RS-STOD}"
if [ ! -d "$RSSTOD_ROOT/images/train" ]; then
  RSSTOD_ROOT="/share/home/u2415363072/5.12/ultralyticsPro--YOLO11/datasets/RS-STOD"
fi
if [ ! -d "$RSSTOD_ROOT/images/train" ]; then
  RSSTOD_ROOT="/share/home/u2415363072/5.3/ultralyticsPro--YOLO11/datasets/RS-STOD"
fi
if [ ! -d "$RSSTOD_ROOT/images/train" ]; then
  RSSTOD_ROOT="/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/datasets/RS-STOD"
fi

test -f "$MODEL_CFG" || { echo "缺少模型配置: $MODEL_CFG"; exit 1; }
test -f "$WEIGHTS" || { echo "缺少初始权重: $WEIGHTS"; exit 1; }

DATA_YAML="$PROJECT_DIR/data_RS_STOD_5p13_${TAG}_e300_${JOB_ID}.yaml"
cat > "$DATA_YAML" <<EOF
path: $RSSTOD_ROOT
train: images/train
val: images/val
test: images/test
nc: 5
names: ['Small Vehicle', 'Large Vehicle', 'Ship', 'Airplane', 'Storage Tank']
EOF

# rm -f "$RSSTOD_ROOT/labels/train.cache" "$RSSTOD_ROOT/labels/val.cache" "$RSSTOD_ROOT/labels/test.cache"

echo "JOB_ID=$JOB_ID"
echo "TAG=$TAG"
echo "MODEL_CFG=$MODEL_CFG"
echo "RSSTOD_ROOT=$RSSTOD_ROOT"
echo "OUTPUT=runs/rsstod_lited1r2_ablation_e300/rsstod_${TAG}_b6_e300_${JOB_ID}"

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
  --project runs/rsstod_lited1r2_ablation_e300 \
  --name "rsstod_${TAG}_b6_e300_${JOB_ID}" \
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
