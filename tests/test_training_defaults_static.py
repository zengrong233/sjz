from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_MODEL = "YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml"


def test_bbox_loss_preserves_configured_iou_loss_branch():
    source = (ROOT / "ultralytics" / "utils" / "loss.py").read_text(encoding="utf-8")
    tail = source.split("Focal Loss改进各类Loss", 1)[1].split("# DFL loss", 1)[0]

    assert "elif useloss == 'NWD':" in source
    assert "loss_iou =" not in tail


def test_training_defaults_use_nosac_fullc2f_main_structure():
    entry = (ROOT / "train_yolo111 copy.py").read_text(encoding="utf-8")
    local = (ROOT / "train_local.sh").read_text(encoding="utf-8")
    slurm = (ROOT / "train_slurm_ab.sh").read_text(encoding="utf-8")

    assert f"default=r'{MAIN_MODEL}'" in entry
    assert f'YAML_PRR_V3="{MAIN_MODEL}"' in local
    assert f'YAML_PRR_V3="{MAIN_MODEL}"' in slurm


def test_training_entry_forwards_spatial_augmentation_overrides():
    entry = (ROOT / "train_yolo111 copy.py").read_text(encoding="utf-8")

    expected = {
        "mosaic": "1.0",
        "scale": "0.5",
        "translate": "0.1",
    }
    for name, default in expected.items():
        assert f"parser.add_argument('--{name}', type=float, default={default}," in entry
        assert entry.count(f'"{name}": opt.{name}') == 2


def test_usod_mosaic_ablation_does_not_require_persistent_cache():
    script = (ROOT / "train_usod_lited1r2_clean_aug_ablation_5p13_slurm.sh").read_text(encoding="utf-8")

    assert "rm -f" not in script
    assert "U1a 提交前要求 train/val cache 已存在" not in script
    assert "cache 不存在，训练进程将自行扫描标签" in script
