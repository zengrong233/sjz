from pathlib import Path
from ultralytics import YOLO


PROJECT_DIR = Path("/share/home/u2415363072/4.25/ultralyticsPro--YOLO11")

WEIGHTS = Path("/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/best_pt/yolo11n.pt")
DATA = PROJECT_DIR / "data_RS_STOD_4p25_test.yaml"

PROJECT_OUT = PROJECT_DIR / "runs/rsstod_test"
RUN_NAME = "rsstod_best_real_test"

if not WEIGHTS.exists():
    raise FileNotFoundError(f"weights not found: {WEIGHTS}")

if not DATA.exists():
    raise FileNotFoundError(f"data yaml not found: {DATA}")

model = YOLO(str(WEIGHTS))

metrics = model.val(
    data=str(DATA),
    split="test",
    imgsz=640,
    batch=4,
    device=0,
    workers=8,
    conf=0.001,
    iou=0.7,
    max_det=300,
    plots=True,
    save_json=True,
    save_txt=True,
    save_conf=True,
    project=str(PROJECT_OUT),
    name=RUN_NAME,
    exist_ok=False,
    verbose=True,
)

print("========== RS-STOD REAL TEST METRICS ==========")
print(f"mAP50-95: {metrics.box.map:.6f}")
print(f"mAP50:    {metrics.box.map50:.6f}")
print(f"mAP75:    {metrics.box.map75:.6f}")

print("per-class mAP50-95:")
for i, v in enumerate(metrics.box.maps):
    print(f"  class {i}: {v:.6f}")

print(f"results saved to: {PROJECT_OUT / RUN_NAME}")
