from pathlib import Path
import os
import math

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SPECS = {
    "rsstod": {
        "src_candidates": ["datasets/RS-STOD"],
        "dst": "datasets/RS-STOD-PREP",
        "yaml": "data_RS-STOD_PREP_5p13.yaml",
        "nc": 5,
        "names": ["Small Vehicle", "Large Vehicle", "Ship", "Airplane", "Storage Tank"],
    },
    "usod": {
        "src_candidates": ["datasets/USOD"],
        "dst": "datasets/USOD-PREP",
        "yaml": "data_USOD_PREP_5p13.yaml",
        "nc": 1,
        "names": ["vehicle"],
    },
    "nwpu": {
        "src_candidates": [
            "datasets/NWPU-VHR10",
            "datasets/NWPU_VHR10",
            "datasets/NWPU VHR-10 v2.v1i.yolov11",
        ],
        "dst": "datasets/NWPU-VHR10-PREP",
        "yaml": "data_NWPU-VHR10_PREP_5p13.yaml",
        "nc": 10,
        "names": [
            "airplane", "baseball diamond", "basketball court", "bridge",
            "ground track field", "harbor", "ship", "storage tank",
            "tennis court", "vehicle",
        ],
    },
}

def pick_root(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p.resolve()
    raise SystemExit(f"[ERR] 找不到数据源: {candidates}")

def find_split(src, split):
    aliases = {
        "train": ["train"],
        "val": ["val", "valid"],
        "test": ["test"],
    }[split]

    checks = []
    for a in aliases:
        checks += [
            (src / "images" / a, src / "labels" / a),
            (src / a / "images", src / a / "labels"),
        ]

    for img_dir, lab_dir in checks:
        if img_dir.exists() and lab_dir.exists():
            return img_dir.resolve(), lab_dir.resolve()
    return None, None

def safe_link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    if not dst.exists():
        os.symlink(src, dst, target_is_directory=True)

def clean_line(line, nc):
    parts = line.strip().split()
    if not parts:
        return None, False
    if len(parts) != 5:
        return None, True

    try:
        cls_f = float(parts[0])
        cls = int(cls_f)
        x, y, w, h = map(float, parts[1:])
    except Exception:
        return None, True

    vals = [x, y, w, h]
    if cls != cls_f or cls < 0 or cls >= nc:
        return None, True
    if any(not math.isfinite(v) for v in vals):
        return None, True
    if w <= 0 or h <= 0:
        return None, True

    # 只修正极小浮点越界；明显错误直接丢弃。
    if any(v < -1e-3 or v > 1 + 1e-3 for v in vals):
        return None, True

    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 1e-6), 1.0)
    h = min(max(h, 1e-6), 1.0)
    return f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}", False

def process_dataset(key, spec):
    src = pick_root(spec["src_candidates"])
    dst = Path(spec["dst"]).resolve()
    nc = spec["nc"]

    print(f"\n[DATASET] {key}")
    print(f"[SRC] {src}")
    print(f"[DST] {dst}")

    total_images = 0
    total_boxes = 0
    dropped = 0

    for split in ["train", "val", "test"]:
        img_dir, lab_dir = find_split(src, split)
        if img_dir is None:
            if split == "test":
                continue
            raise SystemExit(f"[ERR] {key} 缺少 {split} 图像/标签目录")

        safe_link(img_dir, dst / "images" / split)
        out_lab_dir = dst / "labels" / split
        out_lab_dir.mkdir(parents=True, exist_ok=True)

        for cache in out_lab_dir.glob("*.cache*"):
            cache.unlink()

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        split_boxes = 0
        split_drop = 0

        for img in images:
            src_lab = lab_dir / f"{img.stem}.txt"
            out_lab = out_lab_dir / f"{img.stem}.txt"
            lines_out = []

            if src_lab.exists():
                for line in src_lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                    cleaned, bad = clean_line(line, nc)
                    if cleaned is not None:
                        lines_out.append(cleaned)
                    if bad:
                        split_drop += 1

            out_lab.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
            split_boxes += len(lines_out)

        total_images += len(images)
        total_boxes += split_boxes
        dropped += split_drop
        print(f"[{split}] images={len(images)} boxes={split_boxes} dropped={split_drop}")

    yaml_text = (
        f"path: {dst}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {nc}\n"
        f"names: {spec['names']}\n"
    )
    Path(spec["yaml"]).write_text(yaml_text, encoding="utf-8")
    print(f"[OK] yaml={spec['yaml']} images={total_images} boxes={total_boxes} dropped={dropped}")

for key, spec in SPECS.items():
    process_dataset(key, spec)
