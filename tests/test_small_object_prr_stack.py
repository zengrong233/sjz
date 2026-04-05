from pathlib import Path

import pytest
import torch

from ultralytics.engine.batch_curriculum import BatchCurriculumScheduler
from ultralytics.engine.scale_supervision import ScaleSpecificReweighter
from ultralytics.engine.small_object_trainer import SmallObjectABTrainer
from ultralytics.nn.modules.Head.AsDDet import AsDDet
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
PRR_V3_CFG = ROOT / "YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml"


def test_prr_v3_builds_with_asddet_head():
    """PRR-v3 should build with AsDDet instead of silently falling back to Detect."""
    model = DetectionModel(cfg=str(PRR_V3_CFG), nc=10, verbose=False)
    head = model.model[-1]

    assert isinstance(head, AsDDet)
    assert head.nl == 3
    assert tuple(int(s) for s in head.stride.tolist()) == (4, 8, 16)


def test_ssds_reweights_tiny_and_small_targets():
    """SSDS should upweight tiny targets on P2 and small targets on P3."""
    reweighter = ScaleSpecificReweighter(enabled=True)
    target_scores = torch.ones(1, 3, 1)
    target_bboxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 24.0, 24.0], [0.0, 0.0, 80.0, 80.0]]], dtype=torch.float32
    )
    fg_mask = torch.tensor([[True, True, True]])
    stride_per_anchor = torch.tensor([[4.0], [8.0], [16.0]], dtype=torch.float32)

    weighted = reweighter.reweight(target_scores, target_bboxes, fg_mask, [1, 1, 1], stride_per_anchor)

    assert weighted[0, 0, 0].item() == pytest.approx(1.5)
    assert weighted[0, 1, 0].item() == pytest.approx(1.3)
    assert weighted[0, 2, 0].item() == pytest.approx(1.0)
    assert reweighter.stats["tiny_p2_ratio"] == pytest.approx(1.0)
    assert reweighter.stats["small_p3_ratio"] == pytest.approx(1.0)


def test_batch_curriculum_uses_gentle_scales_and_cap():
    """Curriculum should use gentle scales instead of jumping directly to 2x/3x."""
    scheduler = BatchCurriculumScheduler(
        base_accum=11,
        alpha=1.0,
        beta=0.0,
        ema_momentum=0.0,
        warmup_epochs=0,
        medium_accum_scale=1.25,
        hard_accum_scale=1.5,
        max_accum=16,
    )
    medium_batch = {
        "bboxes": torch.tensor(
            [
                [0.0, 0.0, 0.01, 0.01],
                [0.0, 0.0, 0.01, 0.01],
                [0.0, 0.0, 0.20, 0.20],
                [0.0, 0.0, 0.20, 0.20],
                [0.0, 0.0, 0.20, 0.20],
            ],
            dtype=torch.float32,
        ),
        "batch_idx": torch.zeros(5, dtype=torch.int64),
        "img": torch.zeros(1, 3, 640, 640),
    }
    hard_batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 0.01, 0.01]] * 8, dtype=torch.float32),
        "batch_idx": torch.zeros(8, dtype=torch.int64),
        "img": torch.zeros(1, 3, 640, 640),
    }

    assert scheduler.step(medium_batch, epoch=5, imgsz=640) == 14
    assert scheduler.step(hard_batch, epoch=5, imgsz=640) == 16


def test_trainer_syncs_weight_decay_with_dynamic_accumulate():
    """Dynamic accumulate changes should keep optimizer weight_decay in sync."""
    trainer = SmallObjectABTrainer.__new__(SmallObjectABTrainer)
    trainer.ab_cfg = {"base_accum": 12}

    assert trainer._resolve_curriculum_base_accum(8) == 12
    trainer.ab_cfg = {"base_accum": 0}
    assert trainer._resolve_curriculum_base_accum(8) == 8

    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    trainer.optimizer = torch.optim.AdamW(
        [
            {"params": [p1], "weight_decay": 0.01, "group_name": "backbone_decay"},
            {"params": [p2], "weight_decay": 0.0, "group_name": "backbone_no_decay"},
        ],
        lr=0.001,
    )
    trainer._initial_group_weight_decays = [0.01, 0.0]
    trainer._weight_decay_ref_accum = 8
    trainer.accumulate = 12

    trainer._sync_optimizer_weight_decay()

    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.015)
    assert trainer.optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)
