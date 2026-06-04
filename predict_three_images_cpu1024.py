"""CPU + imgsz=1024 高精度对比版本.

8GB 显存跑 1024 + Attention 会 OOM, 所以用 CPU 拼算力换精度.
"""
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHT = PROJECT_ROOT / "best_pt" / "rsstod_best.pt"

SRC_DIR = Path(r"C:\Users\86155\Desktop\水经注\论文\pictures\检测图片")
DST_DIR = Path(r"C:\Users\86155\Desktop\水经注\论文\pictures\检测图片_yolo_out")
DST_DIR.mkdir(parents=True, exist_ok=True)

IMG_NAMES = ["303.jpg", "2298.jpg", "2299.jpg"]


def main() -> None:
    assert WEIGHT.exists(), f"权重不存在: {WEIGHT}"
    model = YOLO(str(WEIGHT))
    print(f"[info] loaded weight: {WEIGHT}")

    sources = [str(SRC_DIR / n) for n in IMG_NAMES]
    results = model.predict(
        source=sources,
        imgsz=1024,
        conf=0.25,
        iou=0.5,
        device="cpu",
        max_det=2000,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(DST_DIR),
        name="rsstod_lited1r2_85036_cpu_imgsz1024",
        exist_ok=True,
        line_width=2,
        verbose=True,
    )

    for r in results:
        n = 0 if r.boxes is None else len(r.boxes)
        cls_counts = {}
        if r.boxes is not None:
            for c in r.boxes.cls.tolist():
                k = model.names[int(c)]
                cls_counts[k] = cls_counts.get(k, 0) + 1
        print(f"[result] {Path(r.path).name}: total={n}  detail={cls_counts}")

    out_run_dir = DST_DIR / "rsstod_lited1r2_85036_cpu_imgsz1024"
    print(f"[done] saved to: {out_run_dir}")


if __name__ == "__main__":
    main()
