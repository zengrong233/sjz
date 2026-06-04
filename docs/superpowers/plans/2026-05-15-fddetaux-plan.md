# FDDetAux Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an experimental FD-style feature distribution auxiliary loss for P2/P3 small-object detection features without changing the E1/CIoU baseline path.

**Architecture:** Keep box/cls/dfl loss unchanged. When YAML enables `fd_aux: true`, `v8DetectionLoss` computes an auxiliary Fréchet-style loss between P2/P3 positive anchor feature distributions and GT-center sampled feature distributions. The first version uses current detection-head feature maps to minimize code risk and make dry-run possible on 5.12.

**Tech Stack:** PyTorch, existing Ultralytics YOLO11 loss path, YAML-driven experiment flags.

---

### Task 1: Add FDDetAux loss helpers

**Files:**
- Modify: `ultralytics/utils/loss.py`

- [ ] Add YAML-gated helper methods to `v8DetectionLoss`.
- [ ] Keep default disabled when `fd_aux` is absent or false.
- [ ] Use P2/P3 only by default.
- [ ] Clamp and detach covariance reference enough to avoid NaN while retaining gradients to current features.

### Task 2: Add YAML branch

**Files:**
- Create: `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2-CIoU-FDDetAux.yaml`

- [ ] Copy LiteD1R2 structure.
- [ ] Set `loss: CIoU`.
- [ ] Add `fd_aux: true`, `fd_aux_weight: 0.005`, `fd_aux_levels: [0, 1]`.

### Task 3: Add 5.12 scripts

**Files:**
- Create: `dryrun_lited1r2_ciou_fddetaux_5p12.sh`
- Create: `train_rsstod_lited1r2_ciou_fddetaux_e100_5p12_slurm.sh`

- [ ] Dry-run compiles Python files and runs `YOLO(...).info()`.
- [ ] Smoke script uses RS-STOD 100 epoch, batch 4 by default, no AMP.

### Task 4: Verification

- [ ] Run `python -m py_compile ultralytics/utils/loss.py` locally.
- [ ] Run YAML existence checks locally.
- [ ] State that model instantiation is deferred to 5.12 because local Windows torch may fail.
