from pathlib import Path
import csv
from ultralytics import YOLO

weights = "/share/home/u2415363072/5.13/ultralyticsPro--YOLO11/best_pt/85210usod_best.pt"
data = "/share/home/u2415363072/5.13/ultralyticsPro--YOLO11/data_USOD_PREP_5p13.yaml"

out_csv = Path("runs/usod_nms_sweep/usod_nms_sweep_summary.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

configs = []

# Hard-NMS: iou_thres 有效
for iou in [0.50, 0.60, 0.70, 0.80, 0.90]:
    configs.append({
        "tag": f"hard_iou{iou:.2f}".replace(".", ""),
        "soft_nms": False,
        "sigma": 0.5,
        "conf": 0.001,
        "iou": iou,
    })

# Soft-NMS: iou_thres 在当前实现中无效，只 sweep sigma
for sigma in [0.30, 0.50, 0.55, 0.60, 0.65, 0.70]:
    configs.append({
        "tag": f"soft_sigma{sigma:.2f}".replace(".", ""),
        "soft_nms": True,
        "sigma": sigma,
        "conf": 0.001,
        "iou": 0.70,
    })

rows = []
for cfg in configs:
    print("=" * 80)
    print(cfg)

    metrics = YOLO(weights).val(
        data=data,
        imgsz=640,
        batch=4,
        device=0,
        workers=8,
        project="runs/usod_nms_sweep",
        name=cfg["tag"],
        exist_ok=True,
        plots=False,
        conf=cfg["conf"],
        iou=cfg["iou"],
        soft_nms=cfg["soft_nms"],
        soft_nms_sigma=cfg["sigma"],
    )

    row = {
        **cfg,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    row["pass_gate"] = (
        row["map50_95"] >= 0.33649 and row["precision"] >= 0.84000
    )
    rows.append(row)
    print(row)

with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] summary={out_csv}")
