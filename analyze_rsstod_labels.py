from pathlib import Path
from collections import Counter, defaultdict
import math

ROOT = Path("/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/datasets/RS-STOD/labels")

names = {
    0: "Small Vehicle",
    1: "Large Vehicle",
    2: "Ship",
    3: "Airplane",
    4: "Storage Tank",
}

bins = {
    "tiny_0_16": lambda a: a <= 16 * 16,
    "small_16_32": lambda a: 16 * 16 < a <= 32 * 32,
    "medium_32_96": lambda a: 32 * 32 < a <= 96 * 96,
    "large_96_plus": lambda a: a > 96 * 96,
}

for split in ["train", "val"]:
    label_dir = ROOT / split
    cls_count = Counter()
    area_bins = defaultdict(Counter)
    wh_stats = defaultdict(list)

    for txt in label_dir.glob("*.txt"):
        for line in txt.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            w = float(parts[3]) * 640
            h = float(parts[4]) * 640
            area = w * h

            cls_count[cls] += 1
            wh_stats[cls].append((w, h, area))

            for bin_name, fn in bins.items():
                if fn(area):
                    area_bins[cls][bin_name] += 1
                    break

    print(f"\n=== {split} ===")
    total = sum(cls_count.values())
    print(f"total boxes: {total}")

    for cls, count in sorted(cls_count.items()):
        arr = wh_stats[cls]
        ws = sorted(x[0] for x in arr)
        hs = sorted(x[1] for x in arr)
        areas = sorted(x[2] for x in arr)

        def q(xs, p):
            if not xs:
                return 0
            return xs[min(len(xs) - 1, int(len(xs) * p))]

        print(f"\n[{cls}] {names.get(cls, cls)}")
        print(f"count={count}, ratio={count / total:.4f}")
        print(f"w: p50={q(ws,0.5):.1f}, p75={q(ws,0.75):.1f}, p90={q(ws,0.9):.1f}")
        print(f"h: p50={q(hs,0.5):.1f}, p75={q(hs,0.75):.1f}, p90={q(hs,0.9):.1f}")
        print(f"area: p50={q(areas,0.5):.1f}, p75={q(areas,0.75):.1f}, p90={q(areas,0.9):.1f}")

        for bin_name in bins:
            n = area_bins[cls][bin_name]
            print(f"{bin_name}: {n} ({n / count:.4f})")