# Ultralytics YOLO 🚀, AGPL-3.0 license
# P2/P3-guided Refocus Resampling（PRR）模块
# 结构主创新：基于 P2/P3 高分辨率分支生成候选区域热力图，再做轻量局部重聚焦增强

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateHeatmapHead(nn.Module):
    """
    候选区域热力图生成头。
    从输入特征生成 1-channel 的小目标候选区域热力图（soft mask）。

    Args:
        in_channels: 输入通道数
        mid_channels: 中间通道数（默认为 in_channels // 4）
    """

    def __init__(self, in_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or max(in_channels // 4, 8)
        # 轻量卷积头：depthwise + pointwise + 1x1 输出
        self.dw = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)
        self.out = nn.Conv2d(mid_channels, 1, 1, bias=True)
        # 初始化 bias 为负值，使初始 sigmoid 输出偏低（稀疏先验）
        nn.init.constant_(self.out.bias, -2.0)

    def forward(self, x):
        """返回 sigmoid 归一化的候选区域热力图 (B, 1, H, W)。"""
        h = self.act(self.bn1(self.dw(x)))
        h = self.act(self.bn2(self.pw(h)))
        return torch.sigmoid(self.out(h))


class SoftRefocusEnhancer(nn.Module):
    """
    软重聚焦增强器（默认实现）。
    使用 candidate heatmap 对原特征做位置加权，再通过轻量卷积做局部增强。

    Args:
        channels: 特征通道数
        gamma_init: 残差系数初始值
    """

    def __init__(self, channels, gamma_init=0.1):
        super().__init__()
        # 局部增强：depthwise conv + pointwise conv
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        # 可学习残差系数
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))

    def forward(self, x, heatmap):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            heatmap: 候选区域热力图 (B, 1, H, W)，值域 [0, 1]

        Returns:
            增强后的特征 (B, C, H, W)
        """
        # 位置加权：高响应区域被放大
        weighted = x * heatmap
        # 局部增强
        enhanced = self.enhance(weighted)
        # 残差回注
        return x + self.gamma * enhanced


class GridSampleRefiner(nn.Module):
    """
    基于 grid_sample 的局部重采样增强器（可选实现）。
    从高响应位置做可学习偏移的重采样，再聚合回原特征。

    Args:
        channels: 特征通道数
        gamma_init: 残差系数初始值
    """

    def __init__(self, channels, gamma_init=0.1):
        super().__init__()
        # 偏移预测：预测 2-channel 的采样偏移
        self.offset_pred = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, 2, 1, bias=True),
        )
        # 偏移初始化为零（初始行为等价于恒等映射）
        nn.init.zeros_(self.offset_pred[-1].weight)
        nn.init.zeros_(self.offset_pred[-1].bias)
        # 增强卷积
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))

    def forward(self, x, heatmap):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            heatmap: 候选区域热力图 (B, 1, H, W)

        Returns:
            增强后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 拼接特征和热力图预测偏移
        offset = self.offset_pred(torch.cat([x, heatmap], dim=1))  # (B, 2, H, W)
        # 构建基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        # 加上偏移（缩放到合理范围）
        offset = offset.permute(0, 2, 3, 1) * 0.1  # (B, H, W, 2)，乘以小系数限制偏移幅度
        # 用热力图加权偏移：只在高响应区域产生有效偏移
        offset = offset * heatmap.permute(0, 2, 3, 1)
        sample_grid = base_grid + offset
        sample_grid = sample_grid.clamp(-1, 1)
        # grid_sample 重采样
        resampled = F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
        enhanced = self.enhance(resampled)
        return x + self.gamma * enhanced


class P2P3RefocusResample(nn.Module):
    """
    P2/P3 引导的重聚焦重采样模块（PRR 主模块）。
    接收 P2 和 P3 特征，分别生成候选热力图并做局部增强。

    在 YAML 中作为独立模块使用，接收 [P2_feat, P3_feat] 双输入。

    Args:
        c_p2: P2 通道数
        c_p3: P3 通道数
        mode: 'softmask' | 'gridsample'，默认 softmask
        gamma_init: 残差系数初始值
    """

    def __init__(self, c_p2, c_p3, mode='softmask', gamma_init=0.1):
        super().__init__()
        self.mode = mode

        # P2 分支
        self.heatmap_p2 = CandidateHeatmapHead(c_p2)
        if mode == 'gridsample':
            self.refocus_p2 = GridSampleRefiner(c_p2, gamma_init)
        else:
            self.refocus_p2 = SoftRefocusEnhancer(c_p2, gamma_init)

        # P3 分支
        self.heatmap_p3 = CandidateHeatmapHead(c_p3)
        if mode == 'gridsample':
            self.refocus_p3 = GridSampleRefiner(c_p3, gamma_init)
        else:
            self.refocus_p3 = SoftRefocusEnhancer(c_p3, gamma_init)

        # 输出通道数（供 tasks.py 解析使用）
        self.c = c_p2  # 主输出通道数取 P2

    def forward(self, x):
        """
        Args:
            x: [P2_feat, P3_feat] 列表

        Returns:
            [enhanced_P2, enhanced_P3] 列表
        """
        p2, p3 = x

        # P2 候选区域 + 重聚焦
        hm_p2 = self.heatmap_p2(p2)
        p2_out = self.refocus_p2(p2, hm_p2)

        # P3 候选区域 + 重聚焦
        hm_p3 = self.heatmap_p3(p3)
        p3_out = self.refocus_p3(p3, hm_p3)

        return [p2_out, p3_out]


class RefocusSingle(nn.Module):
    """
    单分支重聚焦模块（用于 YAML 中对单个特征做 PRR）。
    比 P2P3RefocusResample 更灵活，可以单独插入到任意位置。

    Args:
        channels: 输入/输出通道数
        mode: 'softmask' | 'gridsample'
        gamma_init: 残差系数初始值
    """

    def __init__(self, channels, mode='softmask', gamma_init=0.1):
        super().__init__()
        self.heatmap = CandidateHeatmapHead(channels)
        if mode == 'gridsample':
            self.refocus = GridSampleRefiner(channels, gamma_init)
        else:
            self.refocus = SoftRefocusEnhancer(channels, gamma_init)
        self.c = channels  # 输出通道数

    def forward(self, x):
        """单输入单输出的重聚焦增强。"""
        hm = self.heatmap(x)
        return self.refocus(x, hm)
