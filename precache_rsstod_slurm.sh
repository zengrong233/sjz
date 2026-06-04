#!/bin/bash
#SBATCH -J precache_rsstod
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/precache_%j.out
#SBATCH -e logs/precache_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="/share/home/u2415363072/5.13/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"
mkdir -p logs runs

DATA_YAML="$PROJECT_DIR/data_RS_STOD_precache.yaml"
cat > "$DATA_YAML" <<EOF
path: $PROJECT_DIR/datasets/RS-STOD
train: images/train
val: images/val
test: images/test
nc: 5
names: ['Small Vehicle', 'Large Vehicle', 'Ship', 'Airplane', 'Storage Tank']
EOF

python "train_yolo111 copy.py" \
  --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2-NoMFFF.yaml \
  --weights best_pt/yolo11n.pt \
  --data "$DATA_YAML" \
  --device 0 \
  --epochs 1 \
  --patience 1 \
  --imgsz 640 \
  --batch 1 \
  --workers 8 \
  --project runs/precache_rsstod \
  --name precache \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft
