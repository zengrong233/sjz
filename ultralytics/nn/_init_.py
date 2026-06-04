# Ultralytics YOLO11 🚀
"""
Ultralytics modules.

Example:
    Visualize a module.
"""
# ------------------------------------------------------------------------------------------------------------------------
# Custom module registry for the main-line detector
# (YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-...-LiteD1R2 and its ablations).
# Star-imported by ultralytics/nn/tasks.py so parse_model() can resolve these names.
from ultralytics.nn.core11.EfficientRep import RepVGGBlock, SimConv, RepBlock, Transpose

from ultralytics.nn.core11.GDM import (LAF_px, low_FAM, LAF_h, low_IFM, InjectionMultiSum_Auto_pool1,
InjectionMultiSum_Auto_pool2, InjectionMultiSum_Auto_pool3, InjectionMultiSum_Auto_pool4,
PyramidPoolAgg, TopBasicLayer)

from ultralytics.nn.core11.RepFPN import SimSPPF

from ultralytics.nn.core11.Dysample import DySample
