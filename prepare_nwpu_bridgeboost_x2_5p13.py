from pathlib import Path
import os
import shutil

project = Path("/share/home/u2415363072/5.13/ultralyticsPro--YOLO11")
src = project / "datasets" / "NWPU-VHR10-PREP"
dst = project / "datasets" / "NWPU-VHR10-PREP-BridgeBoost-x2"

bridge_cls = 3
suffix = "__bridgeboost_x2"

def stats(root: Path, split: str):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = [p for p in image_dir.glob("*") if p.is_file()]
    labels = list(label_dir.glob("*.txt"))
    bridge_images = 0
    bridge_boxes = 0

    for label in labels:
        has_bridge = False
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if int(line.split()[0]) == bridge_cls:
                bridge_boxes += 1
                has_bridge = True
        bridge_images += int(has_bridge)

    return len(images), len(labels), bridge_images, bridge_boxes

def find_image(image_dir: Path, stem: str):
    matches = [p for p in image_dir.glob(f"{stem}.*") if p.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"image mapping error: stem={stem}, matches={matches}")
    return matches[0]

def link_or_copy(src_path: Path, dst_path: Path):
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)

if not (src / "images" / "train").is_dir():
    raise SystemExit(f"[ERROR] missing source dataset: {src}")
if not (dst / "images" / "train").is_dir():
    raise SystemExit(f"[ERROR] missing copied dataset: {dst}")

print("=== Baseline copy check ===")
for split in ["train", "val", "test"]:
    s = stats(src, split)
    d = stats(dst, split)
    print(split, "src=", s, "dst=", d)
    if s != d:
        raise SystemExit(
            f"[ERROR] {split} already differs from source; "
            "do not apply oversampling on a modified directory"
        )

existing = list((dst / "labels" / "train").glob(f"*{suffix}.txt"))
if existing:
    raise SystemExit(f"[ERROR] bridge duplicates already exist: {len(existing)}")

train_image_dir = dst / "images" / "train"
train_label_dir = dst / "labels" / "train"
baseline_labels = sorted(train_label_dir.glob("*.txt"))

added = 0
added_boxes = 0

for label in baseline_labels:
    lines = [
        line for line in label.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bridge_boxes = sum(int(line.split()[0]) == bridge_cls for line in lines)
    if bridge_boxes == 0:
        continue

    image = find_image(train_image_dir, label.stem)
    dup_stem = f"{label.stem}{suffix}"
    dup_image = train_image_dir / f"{dup_stem}{image.suffix}"
    dup_label = train_label_dir / f"{dup_stem}.txt"

    if dup_image.exists() or dup_label.exists():
        raise SystemExit(f"[ERROR] duplicate destination exists: {dup_stem}")

    link_or_copy(image, dup_image)
    shutil.copy2(label, dup_label)
    added += 1
    added_boxes += bridge_boxes

for cache in dst.rglob("*.cache"):
    cache.unlink()
for cache in dst.rglob("*.cache.npy"):
    cache.unlink()

src_train = stats(src, "train")
dst_train = stats(dst, "train")
src_val = stats(src, "val")
dst_val = stats(dst, "val")
src_test = stats(src, "test")
dst_test = stats(dst, "test")

if dst_train[0] != src_train[0] + added:
    raise SystemExit("[ERROR] unexpected train image count after oversampling")
if dst_train[1] != src_train[1] + added:
    raise SystemExit("[ERROR] unexpected train label count after oversampling")
if dst_train[3] != src_train[3] + added_boxes:
    raise SystemExit("[ERROR] unexpected bridge box count after oversampling")
if src_val != dst_val or src_test != dst_test:
    raise SystemExit("[ERROR] val/test changed unexpectedly")

print("=== BridgeBoost-x2 completed ===")
print(f"[OK] added_train_images={added}")
print(f"[OK] added_bridge_boxes={added_boxes}")
print(f"[OK] train_before={src_train}")
print(f"[OK] train_after ={dst_train}")
print(f"[OK] val_unchanged ={dst_val}")
print(f"[OK] test_unchanged={dst_test}")
print(f"[OK] output={dst}")
