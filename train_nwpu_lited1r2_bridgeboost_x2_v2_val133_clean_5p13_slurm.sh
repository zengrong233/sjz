#!/bin/bash
#SBATCH -J nwpu_brx2_133
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="/share/home/u2415363072/5.13/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"
mkdir -p logs runs/nwpu_lited1r2_bridgeboost_x2_v2_val133_clean

JOB_ID="${SLURM_JOB_ID:-manual}"
MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"
DATA_YAML="$PROJECT_DIR/data_NWPU-VHR10_PREP_BridgeBoostX2_v2_val133_5p13.yaml"
RUN_NAME="nwpu_bridgeboostx2_v2_val133_yolo11n_b6_e300_${JOB_ID}"

test -f "$MODEL_CFG" || { echo "错误: 缺少 $MODEL_CFG"; exit 1; }
test -f "$WEIGHTS" || { echo "错误: 缺少 $WEIGHTS"; exit 1; }
test -f "$DATA_YAML" || { echo "错误: 缺少 $DATA_YAML"; exit 1; }

if ! grep -Eq '^loss:[[:space:]]*NWD([[:space:]]|$)' "$MODEL_CFG"; then
  echo "错误: 当前 YAML 不是明确的 loss: NWD"
  exit 1
fi

python - <<'PY'
from pathlib import Path
import yaml

project = Path("/share/home/u2415363072/5.13/ultralyticsPro--YOLO11")
src = project / "datasets" / "NWPU-VHR10-PREP"
data_path = project / "data_NWPU-VHR10_PREP_BridgeBoostX2_v2_val133_5p13.yaml"
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

with data_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

assert Path(data["val"]) == src / "images" / "val"
assert Path(data["test"]) == src / "images" / "test"

val_images = sum(1 for p in (src / "images" / "val").rglob("*")
                 if p.is_file() and p.suffix.lower() in exts)
val_boxes = sum(
    sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    for p in (src / "labels" / "val").rglob("*.txt")
)

if (val_images, val_boxes) != (133, 480):
    raise SystemExit(f"错误: 当前验证口径为 {val_images}/{val_boxes}，不是 133/480。")

print("[OK] validation contract = 133 images / 480 boxes")
PY

echo "实验: NWPU BridgeBoost-x2-v2-val133 clean 300"
echo "模型: $MODEL_CFG (loss=NWD)"
echo "权重: $WEIGHTS"
echo "数据: $DATA_YAML"
echo "输出: runs/nwpu_lited1r2_bridgeboost_x2_v2_val133_clean/$RUN_NAME"

nvidia-smi || true

python -m py_compile \
  "train_yolo111 copy.py" \
  ultralytics/nn/core11/GDM.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python -c "from ultralytics import YOLO; m=YOLO('$MODEL_CFG', task='detect'); m.info()"

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
  --project runs/nwpu_lited1r2_bridgeboost_x2_v2_val133_clean \
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
  --mosaic 1.0 \
  --scale 0.5 \
  --translate 0.1

echo "训练完成，退出码: $?"
