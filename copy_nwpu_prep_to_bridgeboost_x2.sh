#!/bin/bash
set -euo pipefail

PROJECT_DIR="/share/home/u2415363072/5.13/ultralyticsPro--YOLO11"
SRC="$PROJECT_DIR/datasets/NWPU-VHR10-PREP"
DST="$PROJECT_DIR/datasets/NWPU-VHR10-PREP-BridgeBoost-x2"

cd "$PROJECT_DIR"

if [ ! -d "$SRC/images/train" ] || [ ! -d "$SRC/labels/train" ]; then
  echo "错误: 源数据目录不存在或格式不正确: $SRC"
  exit 1
fi

if [ -e "$DST" ]; then
  echo "错误: 目标目录已存在，为避免覆盖已有实验数据而终止:"
  echo "$DST"
  exit 1
fi

mkdir -p "$DST"

rsync -a \
  --exclude='*.cache' \
  --exclude='*.cache.npy' \
  "$SRC/" "$DST/"

python - <<'PY'
from pathlib import Path

src = Path("/share/home/u2415363072/5.13/ultralyticsPro--YOLO11/datasets/NWPU-VHR10-PREP")
dst = Path("/share/home/u2415363072/5.13/ultralyticsPro--YOLO11/datasets/NWPU-VHR10-PREP-BridgeBoost-x2")

bridge_cls = 3

def count_split(root, split):
    images = sorted((root / "images" / split).glob("*"))
    labels = sorted((root / "labels" / split).glob("*.txt"))
    bridge_boxes = 0
    bridge_images = 0

    for label in labels:
        has_bridge = False
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0])
            if cls == bridge_cls:
                bridge_boxes += 1
                has_bridge = True
        if has_bridge:
            bridge_images += 1

    return len(images), len(labels), bridge_images, bridge_boxes

print("============================================")
print("NWPU PREP -> BridgeBoost-x2 基线副本校验")
print("============================================")

for split in ["train", "val", "test"]:
    src_stats = count_split(src, split)
    dst_stats = count_split(dst, split)

    print(
        f"{split}: "
        f"src(images={src_stats[0]}, labels={src_stats[1]}, "
        f"bridge_images={src_stats[2]}, bridge_boxes={src_stats[3]}) | "
        f"dst(images={dst_stats[0]}, labels={dst_stats[1]}, "
        f"bridge_images={dst_stats[2]}, bridge_boxes={dst_stats[3]})"
    )

    if src_stats != dst_stats:
        raise SystemExit(f"[ERROR] {split} 复制后统计不一致")

cache_files = list(dst.rglob("*.cache")) + list(dst.rglob("*.cache.npy"))
if cache_files:
    raise SystemExit(f"[ERROR] 派生目录中仍存在 cache 文件: {cache_files}")

print("[OK] 派生目录复制完成，且当前内容与源 PREP 完全一致")
print("[OK] 未包含 cache；尚未执行 bridge oversampling")
print(f"[OK] output={dst}")
PY
