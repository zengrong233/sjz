#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法示例:

1. 默认参数运行:
   python datasplit.py

2. 指定源目录和输出目录:
   python datasplit.py \
       --source-dir "/root/ultralyticsPro--YOLO11/datasets/all" \
       --output-dir "/root/ultralyticsPro--YOLO11/datasets/1000"

3. 指定训练集抽样数量:
   python datasplit.py \
       --source-dir "/root/ultralyticsPro--YOLO11/datasets/all" \
       --output-dir "/root/ultralyticsPro--YOLO11/datasets/1000" \
       --sample-count 1000

4. 指定随机种子:
   python datasplit.py --sample-count 1000 --seed 42

5. 输出目录已存在时强制覆盖:
   python datasplit.py \
       --source-dir "/root/ultralyticsPro--YOLO11/datasets/all" \
       --output-dir "/root/ultralyticsPro--YOLO11/datasets/1000" \
       --overwrite

说明:
- 默认会从训练集 images/VD_train 和 labels/VD_train 中随机抽取 sample-count 个样本
- 验证集 VD_val 和测试集 VD_test 会按当前目录结构完整复制
- 会先输出源数据集统计，再输出最终提取后的统计
- 默认忽略非图片文件，例如 .json
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 YOLO 数据集中随机抽取训练集子集，并重建 train/val/test 目录结构。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/root/ultralyticsPro--YOLO11/datasets/all"),
        help="源数据集根目录，内部应包含 images 和 labels 两个目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/ultralyticsPro--YOLO11/datasets/1000"),
        help="输出数据集根目录",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1000,
        help="从训练集中随机抽取的样本数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证结果可复现",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="VD_train",
        help="训练集子目录名",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="VD_val",
        help="验证集子目录名",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="VD_test",
        help="测试集子目录名",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，则先删除再重建",
    )
    return parser.parse_args()


def list_files(directory: Path, allowed_exts: set[str] | None = None) -> list[Path]:
    if not directory.exists():
        return []
    files = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if allowed_exts is not None and path.suffix.lower() not in allowed_exts:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def build_stem_map(files: list[Path]) -> dict[str, Path]:
    return {file_path.stem: file_path for file_path in files}


def collect_split_info(source_dir: Path, split_name: str) -> dict:
    images_dir = source_dir / "images" / split_name
    labels_dir = source_dir / "labels" / split_name

    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    image_files = list_files(images_dir, IMAGE_EXTS)
    label_files = list_files(labels_dir, {".txt"})

    image_map = build_stem_map(image_files)
    label_map = build_stem_map(label_files)

    image_stems = set(image_map.keys())
    label_stems = set(label_map.keys())

    paired_stems = sorted(image_stems & label_stems)
    image_only_stems = sorted(image_stems - label_stems)
    label_only_stems = sorted(label_stems - image_stems)

    return {
        "split": split_name,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "image_files": image_files,
        "label_files": label_files,
        "image_map": image_map,
        "label_map": label_map,
        "paired_stems": paired_stems,
        "image_only_stems": image_only_stems,
        "label_only_stems": label_only_stems,
    }


def print_stats(title: str, stats_by_split: dict[str, dict]) -> None:
    print(f"\n{title}")
    print("-" * 90)

    total_images = 0
    total_labels = 0

    for split_name, info in stats_by_split.items():
        image_count = len(info["image_files"])
        label_count = len(info["label_files"])
        paired_count = len(info["paired_stems"])
        missing_labels = len(info["image_only_stems"])
        missing_images = len(info["label_only_stems"])

        total_images += image_count
        total_labels += label_count

        print(
            f"{split_name:<10} "
            f"images={image_count:<5} "
            f"labels={label_count:<5} "
            f"paired={paired_count:<5} "
            f"missing_labels={missing_labels:<5} "
            f"missing_images={missing_images:<5}"
        )

    print("-" * 90)
    print(f"TOTAL      images={total_images} labels={total_labels}")


def prepare_output_dir(output_dir: Path, split_names: list[str], overwrite: bool) -> None:
    if output_dir.exists():
        has_content = any(output_dir.iterdir())
        if has_content:
            if overwrite:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"输出目录已存在且非空: {output_dir}\n如需覆盖，请加 --overwrite"
                )

    for kind in ("images", "labels"):
        for split_name in split_names:
            (output_dir / kind / split_name).mkdir(parents=True, exist_ok=True)


def copy_files(file_list: list[Path], dst_dir: Path) -> int:
    copied = 0
    for src_path in file_list:
        shutil.copy2(src_path, dst_dir / src_path.name)
        copied += 1
    return copied


def copy_train_subset(train_info: dict, output_dir: Path, sample_count: int, seed: int) -> tuple[int, int]:
    paired_stems = train_info["paired_stems"]

    if sample_count <= 0:
        raise ValueError("--sample-count 必须大于 0")

    if sample_count > len(paired_stems):
        raise ValueError(
            f"训练集中可抽样的有标签样本只有 {len(paired_stems)} 个，无法抽取 {sample_count} 个"
        )

    rng = random.Random(seed)
    selected_stems = sorted(rng.sample(paired_stems, sample_count))

    train_split = train_info["split"]
    out_img_dir = output_dir / "images" / train_split
    out_lab_dir = output_dir / "labels" / train_split

    copied_images = 0
    copied_labels = 0

    for stem in selected_stems:
        image_path = train_info["image_map"][stem]
        label_path = train_info["label_map"][stem]

        shutil.copy2(image_path, out_img_dir / image_path.name)
        shutil.copy2(label_path, out_lab_dir / label_path.name)

        copied_images += 1
        copied_labels += 1

    return copied_images, copied_labels


def copy_full_split(split_info: dict, output_dir: Path) -> tuple[int, int]:
    split_name = split_info["split"]
    out_img_dir = output_dir / "images" / split_name
    out_lab_dir = output_dir / "labels" / split_name

    copied_images = copy_files(split_info["image_files"], out_img_dir)
    copied_labels = copy_files(split_info["label_files"], out_lab_dir)

    return copied_images, copied_labels


def main() -> int:
    args = parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    split_names = [args.train_split, args.val_split, args.test_split]

    if source_dir.resolve() == output_dir.resolve():
        print("错误: 源目录和输出目录不能相同。", file=sys.stderr)
        return 1

    try:
        source_stats = {
            split_name: collect_split_info(source_dir, split_name)
            for split_name in split_names
        }
    except Exception as exc:
        print(f"读取源数据集失败: {exc}", file=sys.stderr)
        return 1

    print_stats("源数据集验证结果", source_stats)

    train_available = len(source_stats[args.train_split]["paired_stems"])
    if args.sample_count > train_available:
        print(
            f"\n错误: 训练集可抽样的有标签样本只有 {train_available} 个，不能抽取 {args.sample_count} 个。",
            file=sys.stderr,
        )
        return 1

    try:
        prepare_output_dir(output_dir, split_names, args.overwrite)
    except Exception as exc:
        print(f"准备输出目录失败: {exc}", file=sys.stderr)
        return 1

    try:
        train_images, train_labels = copy_train_subset(
            source_stats[args.train_split],
            output_dir,
            args.sample_count,
            args.seed,
        )
        val_images, val_labels = copy_full_split(source_stats[args.val_split], output_dir)
        test_images, test_labels = copy_full_split(source_stats[args.test_split], output_dir)
    except Exception as exc:
        print(f"复制数据失败: {exc}", file=sys.stderr)
        return 1

    output_stats = {
        split_name: collect_split_info(output_dir, split_name)
        for split_name in split_names
    }

    print("\n提取/复制结果")
    print("-" * 90)
    print(f"{args.train_split:<10} 随机抽取: images={train_images} labels={train_labels}")
    print(f"{args.val_split:<10} 完整复制: images={val_images} labels={val_labels}")
    print(f"{args.test_split:<10} 完整复制: images={test_images} labels={test_labels}")
    print("-" * 90)
    print(f"输出目录: {output_dir}")

    print_stats("输出数据集统计结果", output_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
