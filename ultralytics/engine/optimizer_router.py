# Ultralytics YOLO 🚀, AGPL-3.0 license
# Scale-Routed Optimizer：基于模块类型的参数分组路由
# 为小目标检测模型提供差异化学习率和优化器超参

import torch
import torch.nn as nn
from ultralytics.utils import LOGGER, colorstr

# 小目标敏感模块类名列表（与 YOLO11-HFAMPAN-AsDDet-NWD-SmallObject.yaml 对应）
SMALL_OBJECT_MODULE_CLASSES = {
    "TopBasicLayer",
    "LAF_h",
    "InjectionMultiSum_Auto_pool1",
    "InjectionMultiSum_Auto_pool2",
    "InjectionMultiSum_Auto_pool3",
    "InjectionMultiSum_Auto_pool4",
    "MFFF",
    "FrequencyFocusedDownSampling",
    "SemanticAlignmenCalibration",
    "RepBlock",
    "PyramidPoolAgg",
    "RefocusSingle",
    "P2P3RefocusResample",
    "CandidateHeatmapHead",
    "SoftRefocusEnhancer",
    "GridSampleRefiner",
}

# 检测头模块类名
DET_HEAD_MODULE_CLASSES = {
    "Detect",
    "AsDDet",
}

# backbone 中的典型模块类名（用于明确分类）
BACKBONE_MODULE_CLASSES = {
    "Conv",
    "C3k2",
    "SPPF",
    "C2f",
}


def _classify_top_module(module):
    """
    对模型的顶层子模块（model.0, model.1, ...）进行分类。

    基于模块自身的类名判断，而非路径。这样即使子模块是 Conv2d / BatchNorm2d，
    只要其父级顶层模块是 TopBasicLayer，所有参数都会被归入 small_object 组。

    Args:
        module: 顶层子模块实例

    Returns:
        str: 'det_head' | 'small_object' | 'backbone'
    """
    cls_name = type(module).__name__

    # 检测头优先
    if cls_name in DET_HEAD_MODULE_CLASSES:
        return "det_head"

    # 小目标敏感模块
    if cls_name in SMALL_OBJECT_MODULE_CLASSES:
        return "small_object"

    # 其余归入 backbone（包括 Conv, C3k2, SPPF, C2f, Concat, Upsample 等）
    return "backbone"


def build_routed_param_groups(model, base_lr, base_wd, cfg=None):
    """
    基于顶层模块类型构建三组参数，返回 AdamW 可用的 param_groups 列表。

    核心策略：先对 model.model 的每个顶层子模块（model.0 ~ model.N）做分类，
    然后该子模块下的所有参数（包括深层嵌套的 Conv2d / BN 等）都继承父级分类。

    Args:
        model: YOLO 模型（可能被 DDP 包裹）
        base_lr: 基础学习率
        base_wd: 基础 weight_decay
        cfg: 可选的超参覆盖字典

    Returns:
        list[dict]: param_groups，每组包含 params / lr / betas / weight_decay / group_name
        dict: 路由统计信息
    """
    # 超参默认值（可通过 cfg 覆盖）
    cfg = cfg or {}
    backbone_lr_scale = cfg.get("backbone_lr_scale", 0.5)
    smallobj_lr_scale = cfg.get("smallobj_lr_scale", 1.25)
    head_lr_scale = cfg.get("head_lr_scale", 1.5)
    smallobj_beta2 = cfg.get("smallobj_beta2", 0.9995)
    smallobj_wd_scale = cfg.get("smallobj_wd_scale", 0.5)

    # 解包 DDP
    raw_model = model.module if hasattr(model, "module") else model

    # Norm 层类型集合（bias 和 norm 参数不加 weight_decay）
    bn_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    # 三组容器：每组分 decay / no_decay
    groups = {
        "backbone": {"decay": [], "no_decay": []},
        "small_object": {"decay": [], "no_decay": []},
        "det_head": {"decay": [], "no_decay": []},
    }
    routing_log = {"backbone": [], "small_object": [], "det_head": []}

    # ----------------------------------------------------------------
    # 步骤 1：建立顶层模块索引 -> 分组名称的映射
    # YOLO 模型结构：raw_model.model 是 nn.Sequential，子模块为 model.0, model.1, ...
    # ----------------------------------------------------------------
    top_module_map = {}  # {idx: (group_name, cls_name)}

    # 获取模型的 Sequential 容器
    model_seq = getattr(raw_model, "model", None)
    if model_seq is None:
        # 回退：如果没有 .model 属性，尝试直接遍历
        LOGGER.warning("optimizer_router: 未找到 raw_model.model，回退到逐模块分类")
        return _fallback_build(raw_model, base_lr, base_wd, cfg)

    for idx, top_mod in enumerate(model_seq):
        group_name = _classify_top_module(top_mod)
        cls_name = type(top_mod).__name__
        top_module_map[idx] = (group_name, cls_name)

    # 打印顶层模块映射
    if True:  # 始终打印，方便调试
        LOGGER.info(colorstr("bold", "Scale-Routed Optimizer 顶层模块映射："))
        for idx, (gname, cname) in sorted(top_module_map.items()):
            LOGGER.info(f"  model.{idx:2d} | {cname:40s} -> {gname}")

    # ----------------------------------------------------------------
    # 步骤 2：遍历所有参数，根据路径前缀分配到对应组
    # 参数路径格式：model.{idx}.xxx.yyy.weight
    # ----------------------------------------------------------------
    seen_ids = set()

    for param_name, param in raw_model.named_parameters():
        if id(param) in seen_ids or not param.requires_grad:
            continue
        seen_ids.add(id(param))

        # 解析顶层模块索引
        group_name = "backbone"  # 默认
        parts = param_name.split(".")
        # 期望格式: model.{idx}.xxx... 或直接 {idx}.xxx...
        if len(parts) >= 2:
            # 尝试提取索引
            idx_str = None
            if parts[0] == "model" and len(parts) >= 3:
                idx_str = parts[1]
            elif parts[0].isdigit():
                idx_str = parts[0]

            if idx_str is not None and idx_str.isdigit():
                idx = int(idx_str)
                if idx in top_module_map:
                    group_name = top_module_map[idx][0]

        # 判断是否为 no_decay 参数（bias 或 norm 层参数）
        # 需要找到参数所属的直接模块来判断是否是 norm 层
        is_no_decay = _is_no_decay_param(raw_model, param_name, param, bn_types)

        if is_no_decay:
            groups[group_name]["no_decay"].append(param)
        else:
            groups[group_name]["decay"].append(param)
        routing_log[group_name].append(param_name)

    # ----------------------------------------------------------------
    # 步骤 3：构建 param_groups
    # ----------------------------------------------------------------
    param_groups = []

    # backbone
    if groups["backbone"]["no_decay"]:
        param_groups.append({
            "params": groups["backbone"]["no_decay"],
            "lr": base_lr * backbone_lr_scale,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "group_name": "backbone_no_decay",
        })
    if groups["backbone"]["decay"]:
        param_groups.append({
            "params": groups["backbone"]["decay"],
            "lr": base_lr * backbone_lr_scale,
            "betas": (0.9, 0.999),
            "weight_decay": base_wd,
            "group_name": "backbone_decay",
        })

    # small_object
    if groups["small_object"]["no_decay"]:
        param_groups.append({
            "params": groups["small_object"]["no_decay"],
            "lr": base_lr * smallobj_lr_scale,
            "betas": (0.9, smallobj_beta2),
            "weight_decay": 0.0,
            "group_name": "small_object_no_decay",
        })
    if groups["small_object"]["decay"]:
        param_groups.append({
            "params": groups["small_object"]["decay"],
            "lr": base_lr * smallobj_lr_scale,
            "betas": (0.9, smallobj_beta2),
            "weight_decay": base_wd * smallobj_wd_scale,
            "group_name": "small_object_decay",
        })

    # det_head
    if groups["det_head"]["no_decay"]:
        param_groups.append({
            "params": groups["det_head"]["no_decay"],
            "lr": base_lr * head_lr_scale,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "group_name": "det_head_no_decay",
        })
    if groups["det_head"]["decay"]:
        param_groups.append({
            "params": groups["det_head"]["decay"],
            "lr": base_lr * head_lr_scale,
            "betas": (0.9, 0.999),
            "weight_decay": base_wd,
            "group_name": "det_head_decay",
        })

    # 统计信息
    stats = {}
    for gname in ["backbone", "small_object", "det_head"]:
        n_params = sum(p.numel() for p in groups[gname]["decay"]) + \
                   sum(p.numel() for p in groups[gname]["no_decay"])
        n_tensors = len(groups[gname]["decay"]) + len(groups[gname]["no_decay"])
        samples = routing_log[gname][:5]
        stats[gname] = {"n_params": n_params, "n_tensors": n_tensors, "samples": samples}

    return param_groups, stats


def _is_no_decay_param(model, param_name, param, bn_types):
    """
    判断参数是否应该免除 weight_decay。

    规则：bias 参数 或 归属于 Norm 层的参数 不加 weight_decay。

    Args:
        model: 原始模型
        param_name: 参数完整路径
        param: 参数张量
        bn_types: Norm 层类型元组

    Returns:
        bool: True 表示不加 weight_decay
    """
    # bias 参数
    if param_name.endswith(".bias"):
        return True

    # 检查参数所属的直接父模块是否是 Norm 层
    # 从参数路径中提取模块路径：去掉最后的 .weight / .bias
    parts = param_name.rsplit(".", 1)
    if len(parts) == 2:
        mod_path = parts[0]
        try:
            mod = model
            for attr in mod_path.split("."):
                if attr.isdigit():
                    mod = mod[int(attr)]
                else:
                    mod = getattr(mod, attr)
            if isinstance(mod, bn_types):
                return True
        except (AttributeError, IndexError, TypeError):
            pass

    # 参数维度为 1 的通常是 norm 层参数
    if param.ndim <= 1:
        return True

    return False


def _fallback_build(model, base_lr, base_wd, cfg):
    """
    回退方案：当模型结构不符合预期时，使用逐模块遍历分类。
    所有参数归入 backbone 组。
    """
    cfg = cfg or {}
    backbone_lr_scale = cfg.get("backbone_lr_scale", 0.5)
    bn_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay_param(model, name, param, bn_types):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if no_decay_params:
        param_groups.append({
            "params": no_decay_params,
            "lr": base_lr * backbone_lr_scale,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "group_name": "backbone_no_decay",
        })
    if decay_params:
        param_groups.append({
            "params": decay_params,
            "lr": base_lr * backbone_lr_scale,
            "betas": (0.9, 0.999),
            "weight_decay": base_wd,
            "group_name": "backbone_decay",
        })

    total = sum(p.numel() for p in decay_params) + sum(p.numel() for p in no_decay_params)
    stats = {
        "backbone": {"n_params": total, "n_tensors": len(decay_params) + len(no_decay_params), "samples": []},
        "small_object": {"n_params": 0, "n_tensors": 0, "samples": []},
        "det_head": {"n_params": 0, "n_tensors": 0, "samples": []},
    }
    LOGGER.warning("optimizer_router: 使用回退方案，所有参数归入 backbone 组")
    return param_groups, stats


def print_routing_summary(stats, base_lr, cfg=None):
    """打印参数路由摘要。"""
    cfg = cfg or {}
    backbone_lr_scale = cfg.get("backbone_lr_scale", 0.5)
    smallobj_lr_scale = cfg.get("smallobj_lr_scale", 1.25)
    head_lr_scale = cfg.get("head_lr_scale", 1.5)

    LOGGER.info(colorstr("bold", "Scale-Routed Optimizer 参数路由摘要："))
    for gname, info in stats.items():
        if gname == "backbone":
            lr = base_lr * backbone_lr_scale
        elif gname == "small_object":
            lr = base_lr * smallobj_lr_scale
        else:
            lr = base_lr * head_lr_scale
        LOGGER.info(
            f"  {gname:20s} | 参数量: {info['n_params']:>10,} | 张量数: {info['n_tensors']:>4} | lr: {lr:.6f}"
        )
        if info["samples"]:
            LOGGER.info(f"    样例: {info['samples']}")
    total = sum(s["n_params"] for s in stats.values())
    LOGGER.info(f"  {'总计':20s} | 参数量: {total:>10,}")


def print_model_module_tree(model, max_depth=3):
    """
    调试函数：打印模型模块树，帮助确认模块名与 YAML 配置的对应关系。

    Args:
        model: YOLO 模型
        max_depth: 最大打印深度
    """
    raw_model = model.module if hasattr(model, "module") else model
    LOGGER.info(colorstr("bold", "模型模块树（调试）："))

    model_seq = getattr(raw_model, "model", None)
    if model_seq is not None:
        for idx, top_mod in enumerate(model_seq):
            cls_name = type(top_mod).__name__
            group = _classify_top_module(top_mod)
            n_params = sum(p.numel() for p in top_mod.parameters())
            LOGGER.info(f"  model.{idx:2d} | {cls_name:40s} | {n_params:>10,} params | -> {group}")
            if max_depth > 0:
                for sub_name, sub_mod in top_mod.named_modules():
                    if sub_name == "":
                        continue
                    depth = sub_name.count(".") + 1
                    if depth <= max_depth:
                        sub_cls = type(sub_mod).__name__
                        sub_params = sum(p.numel() for p in sub_mod.parameters(recurse=False))
                        if sub_params > 0:
                            indent = "    " + "  " * depth
                            LOGGER.info(f"{indent}{sub_name:50s} | {sub_cls:25s} | {sub_params:>8,} params")
    else:
        for name, mod in raw_model.named_modules():
            depth = name.count(".")
            if depth <= max_depth:
                cls_name = type(mod).__name__
                n_params = sum(p.numel() for p in mod.parameters(recurse=False))
                if n_params > 0:
                    indent = "  " * depth
                    LOGGER.info(f"  {indent}{name:60s} | {cls_name:30s} | {n_params:>8,} params")
