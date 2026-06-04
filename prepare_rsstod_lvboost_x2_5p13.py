from pathlib import Path
import os
import shutil

src = Path("datasets/RS-STOD")
dst = Path("datasets/RS-STOD-LVBoost-x2")

if dst.exists():
    raise SystemExit(f"[ERROR] output already exists: {dst}")

img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
large_vehicle_cls = "1"

def link_or_copy(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)

def find_image(split: str, stem: str):
    for ext in img_exts:
        p = src / "images" / split / f"{stem}{ext}"
        if p.exists():
            return p
    return None

total_train = 0
lv_train = 0
extra = 0

for split in ["train", "val", "test"]:
    label_dir = src / "labels" / split
    image_dir = src / "images" / split
    if not label_dir.exists():
        continue

    for lab in sorted(label_dir.glob("*.txt")):
        stem = lab.stem
        img = find_image(split, stem)
        if img is None:
            print(f"[WARN] missing image for label: {lab}")
            continue

        link_or_copy(img, dst / "images" / split / img.name)
        link_or_copy(lab, dst / "labels" / split / lab.name)

        if split == "train":
            total_train += 1
            has_lv = any(
                line.strip().split(maxsplit=1)[0] == large_vehicle_cls
                for line in lab.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if has_lv:
                lv_train += 1
                dup_stem = f"{stem}_lvb1"
                link_or_copy(img, dst / "images" / split / f"{dup_stem}{img.suffix}")
                link_or_copy(lab, dst / "labels" / split / f"{dup_stem}.txt")
                extra += 1

print(f"[OK] source={src}")
print(f"[OK] output={dst}")
print(f"[OK] train_images={total_train}, lv_train_images={lv_train}, extra_copies={extra}")
print(f"[OK] lvboost_train_images={total_train + extra}")
