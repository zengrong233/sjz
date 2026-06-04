from ultralytics import YOLO

model = "/share/home/u2415363072/5.13/ultralyticsPro--YOLO11/runs/nwpu_lited1r2_clean/nwpu_clean_yolo11n_b6_e300_87416/weights/best.pt"
data = "/share/home/u2415363072/5.13/ultralyticsPro--YOLO11/data_NWPU-VHR10_Val86_ZIP_5p13.yaml"

m = YOLO(model)
m.val(
    data=data,
    split="test",
    imgsz=640,
    batch=6,
    workers=8,
    device=0,
    project="runs/nwpu_aimlaux_bridge_probe",
    name="87416_test_split_anchor",
    plots=True,
)
