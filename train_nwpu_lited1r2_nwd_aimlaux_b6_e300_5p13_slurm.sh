#!/bin/bash
#SBATCH -J nwpu_aimlaux300
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
mkdir -p logs runs/nwpu_lited1r2_nwd_aimlaux_e300

JOB_ID="${SLURM_JOB_ID:-manual}"

BASE_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2-NWD-AIMLAux.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"

DATA_ROOT="$PROJECT_DIR/datasets/NWPU-VHR10-Original-Val86-ZIP"
DATA_YAML="$PROJECT_DIR/data_NWPU-VHR10_Val86_ZIP_5p13.yaml"

if [ ! -f "$MODEL_CFG" ]; then
python - <<'PY'
from pathlib import Path

src = Path("YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml")
dst = Path("YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2-NWD-AIMLAux.yaml")

s = src.read_text(encoding="utf-8")
if "loss: NWD" not in s:
    raise SystemExit("错误: 源 YAML 不是 loss: NWD")

insert = """loss: NWD

fd_aux: true
fd_aux_weight: 0.003
fd_aux_levels: [0, 1]
fd_aux_area_thr: 1024
fd_aux_min_samples: 8
fd_aux_max_samples: 512
fd_aux_eps: 0.000001
"""
s = s.replace("loss: NWD\n", insert, 1)
dst.write_text(s, encoding="utf-8")
print(f"[OK] created {dst}")
PY
fi

if [ ! -d "$DATA_ROOT/images/train" ] || [ ! -d "$DATA_ROOT/images/val" ]; then
  echo "错误: 未找到 NWPU Val86 ZIP 数据集: $DATA_ROOT"
  exit 1
fi

cat > "$DATA_YAML" <<EOF
path: $DATA_ROOT
train: images/train
val: images/val
test: images/test

nc: 10
names:
  0: airplane
  1: baseball diamond
  2: basketball court
  3: bridge
  4: ground track field
  5: harbor
  6: ship
  7: storage tank
  8: tennis court
  9: vehicle
EOF

DATA_ROOT="$DATA_ROOT" python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["DATA_ROOT"])
img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def count_split(split):
    imgs = [p for p in (root / "images" / split).iterdir() if p.suffix.lower() in img_ext]
    boxes = 0
    bridge = 0
    for lab in (root / "labels" / split).glob("*.txt"):
        for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                boxes += 1
                if parts[0] == "3":
                    bridge += 1
    return len(imgs), boxes, bridge

train = count_split("train")
val = count_split("val")
test = count_split("test")

print(f"[CHECK] train images={train[0]}, boxes={train[1]}")
print(f"[CHECK] val images={val[0]}, boxes={val[1]}, bridge={val[2]}")
print(f"[CHECK] test images={test[0]}, boxes={test[1]}")

if val[0] != 86 or val[1] != 480 or val[2] != 8:
    raise SystemExit("[ERROR] NWPU val 不是 87416 的 Val86/480/bridge8 口径，停止训练。")
PY

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到 yolo11n 初始权重: $WEIGHTS"
  exit 1
fi

if ! grep -qE '^loss:[[:space:]]*NWD' "$MODEL_CFG"; then
  echo "错误: 当前 YAML 不是明确的 loss: NWD"
  grep -nE '^loss:' "$MODEL_CFG" || true
  exit 1
fi

if ! grep -qE '^fd_aux:[[:space:]]*true' "$MODEL_CFG"; then
  echo "错误: 当前 YAML 未开启 fd_aux"
  grep -nE '^fd_aux' "$MODEL_CFG" || true
  exit 1
fi

echo "============================================"
echo "作业ID: $JOB_ID"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "数据 YAML: $DATA_YAML"
echo "数据根目录: $DATA_ROOT"
echo "初始权重: $WEIGHTS"
echo "输出目录: runs/nwpu_lited1r2_nwd_aimlaux_e300/nwpu_nwd_aimlaux_w0003_val86_yolo11n_b6_e300_${JOB_ID}"
echo "loss/fd_aux_weight: NWD / 0.003"
echo "batch/imgsz/amp: 6/640/false"
echo "============================================"

nvidia-smi || true

python -m py_compile \
  "train_yolo111 copy.py" \
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
  --patience 100 \
  --imgsz 640 \
  --batch 6 \
  --workers 8 \
  --project runs/nwpu_lited1r2_nwd_aimlaux_e300 \
  --name "nwpu_nwd_aimlaux_w0003_val86_yolo11n_b6_e300_${JOB_ID}" \
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
