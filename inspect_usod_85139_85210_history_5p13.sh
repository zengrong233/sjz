#!/bin/bash
# Extract metadata for the staged USOD result chain without starting training.

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_5P3="${PROJECT_5P3:-/share/home/u2415363072/5.3/ultralyticsPro--YOLO11}"
PROJECT_5P12="${PROJECT_5P12:-/share/home/u2415363072/5.12/ultralyticsPro--YOLO11}"
PROJECT_5P13="${PROJECT_5P13:-/share/home/u2415363072/5.13/ultralyticsPro--YOLO11}"
OUT_DIR="${OUT_DIR:-$PROJECT_5P13/runs/usod_history_evidence}"

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$OUT_DIR/history_85139_85210_${STAMP}.txt"

export PROJECT_5P3 PROJECT_5P12 PROJECT_5P13 REPORT

python - <<'PY' | tee "$REPORT"
import csv
import os
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(f"错误: 当前环境缺少 PyYAML，无法读取 args.yaml: {exc}")

ids = ("85139", "85210")
roots = [
    Path(os.environ["PROJECT_5P3"]),
    Path(os.environ["PROJECT_5P12"]),
    Path(os.environ["PROJECT_5P13"]),
]
keys = [
    "model",
    "data",
    "epochs",
    "patience",
    "batch",
    "imgsz",
    "pretrained",
    "optimizer",
    "seed",
    "amp",
    "lr0",
    "lrf",
    "warmup_epochs",
    "warmup_bias_lr",
    "mosaic",
    "scale",
    "translate",
    "soft_nms",
    "soft_nms_sigma",
    "max_det",
]


def candidate_artifacts(root: Path):
    if not root.exists():
        print(f"[MISSING ROOT] {root}")
        return [], []
    dirs = []
    logs = []
    for base_name in ("runs", "best_pt", "logs"):
        base = root / base_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if any(job_id in str(path) for job_id in ids):
                dirs.append(path if path.is_dir() else path.parent)
                if path.is_file() and path.suffix.lower() in {".out", ".err", ".log"}:
                    logs.append(path)
    return sorted(set(dirs)), sorted(set(logs))


def best_result(results_file: Path):
    with results_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    map_key = next(
        (key for key in rows[0] if "mAP50-95" in key or "mAP50-95(B)" in key),
        None,
    )
    if map_key is None:
        return {"error": "results.csv 中未找到 mAP50-95 列"}
    best_idx, best_row = max(
        enumerate(rows, start=1),
        key=lambda item: float(item[1][map_key].strip()),
    )
    def metric(fragment):
        key = next((name for name in best_row if fragment in name), None)
        return best_row[key].strip() if key else "<missing>"
    return {
        "epoch_row": best_idx,
        "precision": metric("Precision"),
        "recall": metric("Recall"),
        "map50": metric("mAP50(B)") if any("mAP50(B)" in k for k in best_row) else metric("mAP50"),
        "map50_95": best_row[map_key].strip(),
    }


print("=" * 88)
print("USOD staged-training evidence: jobs 85139 -> 85210")
print("说明: 本脚本只读取元数据，不修改训练结果、不启动 GPU 训练。")
print("=" * 88)

all_hits = []
all_logs = []
for root in roots:
    print(f"\n[SEARCH ROOT] {root}")
    hits, logs = candidate_artifacts(root)
    all_hits.extend(hits)
    all_logs.extend(logs)
    if not hits:
        print("  未发现名称含 85139/85210 的路径")
    else:
        for hit in hits:
            print(f"  {hit}")

run_dirs = []
for path in sorted(set(all_hits)):
    current = path
    while current not in run_dirs and current != current.parent:
        if (current / "args.yaml").is_file() or (current / "results.csv").is_file():
            run_dirs.append(current)
            break
        current = current.parent

print("\n" + "=" * 88)
if not run_dirs:
    print("[RESULT] 未发现可解析的 args.yaml/results.csv。")
    print("请从超算保留目录补回 85139/85210 对应 run，或扩展 PROJECT_5P* 路径。")
else:
    for run_dir in sorted(set(run_dirs)):
        print(f"\n[RUN] {run_dir}")
        args_file = run_dir / "args.yaml"
        results_file = run_dir / "results.csv"
        if args_file.is_file():
            args = yaml.safe_load(args_file.read_text(encoding="utf-8")) or {}
            print("  [ARGS]")
            for key in keys:
                if key in args:
                    print(f"    {key}: {args[key]}")
        else:
            print("  [ARGS] <missing>")
        if results_file.is_file():
            print(f"  [BEST] {best_result(results_file)}")
        else:
            print("  [BEST] <results.csv missing>")

print("\n" + "=" * 88)
print("[LOG EVIDENCE]")
markers = (
    "模型配置",
    "初始权重",
    "数据 YAML",
    "输出目录",
    "epochs",
    "lr0",
    "warmup",
    "mosaic",
    "Validating",
    "Results saved",
    "训练完成",
)
if not all_logs:
    print("  未发现匹配的 .out/.err/.log 文件")
else:
    for log_file in sorted(set(all_logs)):
        print(f"\n  [LOG] {log_file}")
        try:
            matched = [
                line.rstrip()
                for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                if any(marker in line for marker in markers)
            ]
        except OSError as exc:
            print(f"    无法读取: {exc}")
            continue
        if not matched:
            print("    未匹配到关键超参/结果行")
        for line in matched[:80]:
            print(f"    {line}")
        if len(matched) > 80:
            print(f"    ... 另有 {len(matched) - 80} 行已省略")

print("\n" + "=" * 88)
print(f"[REPORT TARGET] {os.environ['REPORT']}")
PY

echo "证据提取报告已写入: $REPORT"
