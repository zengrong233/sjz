"""使用 RS-STOD 主线权重 (85036 LiteD1R2) 对论文用 3 张图做推理并保存带框结果.

输入目录: C:/Users/86155/Desktop/水经注/论文/pictures/检测图片
输出目录: C:/Users/86155/Desktop/水经注/论文/pictures/检测图片_yolo_out

权重: best_pt/rsstod_best.pt  (5 类: Small/Large Vehicle, Ship, Airplane, Storage Tank)
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
    print(f"[info] class names : {model.names}")

    sources = [str(SRC_DIR / n) for n in IMG_NAMES]
    for s in sources:
        assert Path(s).exists(), f"图片不存在: {s}"

    # 8GB 显存上 1024 会 OOM, 先用 640 + GPU; 后面如果想要更精细可切到 CPU 跑 1024.
    results = model.predict(
        source=sources,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device=0,
        max_det=1000,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(DST_DIR),
        name="rsstod_lited1r2_85036_imgsz640",
        exist_ok=True,
        line_width=2,
        verbose=True,
    )

    # 简单打印每张图的检测统计
    for r in results:
        n = 0 if r.boxes is None else len(r.boxes)
        cls_counts = {}
        if r.boxes is not None:
            for c in r.boxes.cls.tolist():
                k = model.names[int(c)]
                cls_counts[k] = cls_counts.get(k, 0) + 1
        print(f"[result] {Path(r.path).name}: total={n}  detail={cls_counts}")

    out_run_dir = DST_DIR / "rsstod_lited1r2_85036_imgsz640"
    print(f"[done] saved to: {out_run_dir}")


if __name__ == "__main__":
    main()
