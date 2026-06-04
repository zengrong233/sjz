#!/bin/bash
#SBATCH -J nwpu_brx2_v86
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="/share/home/u2415363072/5.13/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"
mkdir -p logs runs/nwpu_lited1r2_bridgeboostx2_val86_zip_clean

JOB_ID="${SLURM_JOB_ID:-manual}"
MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
DATA_YAML="$PROJECT_DIR/data_NWPU-VHR10_BridgeBoostX2_Val86_ZIP_5p13.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"

test -f "$MODEL_CFG" || { echo "ERROR: missing model $MODEL_CFG"; exit 1; }
test -f "$DATA_YAML" || { echo "ERROR: missing data yaml $DATA_YAML"; exit 1; }
test -f "$WEIGHTS" || { echo "ERROR: missing weights $WEIGHTS"; exit 1; }

grep -nE '^loss:' "$MODEL_CFG" | grep -q 'NWD' || {
  echo "ERROR: current model is not loss: NWD"
  exit 1
}

python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(
    Path("data_NWPU-VHR10_BridgeBoostX2_Val86_ZIP_5p13.yaml").read_text(
        encoding="utf-8"
    )
)
root = Path(cfg["path"])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def stats(split):
    images = [
        p for p in (root / "images" / split).iterdir()
        if p.suffix.lower() in exts
    ]
    boxes = sum(
        sum(1 for x in p.read_text(encoding="utf-8").splitlines() if x.strip())
        for p in (root / "labels" / split).glob("*.txt")
    )
    return len(images), boxes

train_stat = stats("train")
val_stat = stats("val")
test_stat = stats("test")

print(f"[CHECK] train={train_stat}")
print(f"[CHECK] val={val_stat}")
print(f"[CHECK] test={test_stat}")

assert val_stat == (86, 480), "validation protocol changed"
assert test_stat == (42, 267), "test protocol changed"
assert train_stat[0] > 1044, "bridge oversampling was not applied"
PY

python -m py_compile \
  "train_yolo111 copy.py" \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python -c "from ultralytics import YOLO; YOLO('$MODEL_CFG', task='detect').info()"

echo "============================================"
echo "JOB_ID=$JOB_ID"
echo "MODEL=$MODEL_CFG"
echo "DATA=$DATA_YAML"
echo "WEIGHTS=$WEIGHTS"
echo "PROTOCOL=yolo11n clean + bridge train-only x2 + original val86"
echo "OUTPUT=runs/nwpu_lited1r2_bridgeboostx2_val86_zip_clean/nwpu_bridgeboostx2_val86_zip_yolo11n_b6_e300_${JOB_ID}"
echo "============================================"

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
  --seed 0 \
  --project runs/nwpu_lited1r2_bridgeboostx2_val86_zip_clean \
  --name "nwpu_bridgeboostx2_val86_zip_yolo11n_b6_e300_${JOB_ID}" \
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
  --mosaic 1.0 \
  --scale 0.5 \
  --translate 0.1

echo "训练完成，退出码: $?"
