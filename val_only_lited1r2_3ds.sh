#!/bin/bash
#SBATCH -J val_lited1r2_3ds
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/slurm_val3ds_%j.out
#SBATCH -e logs/slurm_val3ds_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.13/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"
mkdir -p logs runs/val_lited1r2_3ds_prep

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 未找到主线 YAML: $MODEL_CFG"
  exit 1
fi

if ! grep -qE '^loss:[[:space:]]*NWD' "$MODEL_CFG"; then
  echo "错误: 当前主线 YAML 不是明确的 loss: NWD"
  grep -nE '^loss:' "$MODEL_CFG" || true
  exit 1
fi

python -m py_compile \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/modules/block.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python - <<'PY'
import csv
import os
from pathlib import Path
from ultralytics import YOLO

project = Path(os.environ.get("PROJECT_DIR", "/share/home/u2415363072/5.13/ultralyticsPro--YOLO11")).resolve()

specs = [
    {
        "tag": "rsstod",
        "weights": project / "best_pt/85036rsstod_best.pt",
        "data": project / "data_RS-STOD_PREP_5p13.yaml",
        "batch": 6,
        "target_map": 0.45767,
    },
    {
        "tag": "usod",
        "weights": project / "best_pt/85210usod_best.pt",
        "data": project / "data_USOD_PREP_5p13.yaml",
        "batch": 4,
        "target_map": 0.33549,
    },
    {
        "tag": "nwpu",
        "weights": project / "best_pt/86913nwpu_best.pt",
        "data": project / "data_NWPU-VHR10_PREP_5p13.yaml",
        "batch": 6,
        "target_map": 0.53562,
    },
]

for spec in specs:
    if not spec["weights"].is_file():
        raise FileNotFoundError(f"missing weights: {spec['weights']}")
    if not spec["data"].is_file():
        raise FileNotFoundError(f"missing data yaml: {spec['data']}")

rows = []
for spec in specs:
    tag = spec["tag"]
    print("=" * 88)
    print(f"[VAL] {tag}")
    print(f"[WEIGHTS] {spec['weights']}")
    print(f"[DATA] {spec['data']}")
    print(f"[TARGET mAP50-95] {spec['target_map']}")

    # 直接加载 checkpoint，避免用 nc=80 的根 YAML 重建头部造成类别维度不匹配。
    model = YOLO(str(spec["weights"]), task="detect")
    model.info(detailed=False)
    metrics = model.val(
        data=str(spec["data"]),
        imgsz=640,
        batch=spec["batch"],
        device=0,
        workers=8,
        project="runs/val_lited1r2_3ds_prep",
        name=f"val_{tag}",
        split="val",
        plots=True,
        verbose=True,
    )

    box = metrics.box
    row = {
        "dataset": tag,
        "weights": str(spec["weights"]),
        "data": str(spec["data"]),
        "target_map50_95": spec["target_map"],
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "delta_map50_95": float(box.map) - float(spec["target_map"]),
    }
    rows.append(row)
    print(f"[RESULT] {row}")

out = project / "runs/val_lited1r2_3ds_prep/val_only_lited1r2_3ds_summary.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("=" * 88)
print(f"[SUMMARY] {out}")
for row in rows:
    print(
        f"{row['dataset']}: "
        f"P={row['precision']:.5f}, R={row['recall']:.5f}, "
        f"mAP50={row['map50']:.5f}, mAP50-95={row['map50_95']:.5f}, "
        f"delta={row['delta_map50_95']:+.5f}"
    )
PY

echo "val-only 完成，退出码: $?"
