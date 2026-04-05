import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Microsoft YaHei']

plt.rcParams['font.size'] = 12

# 数据
labels = np.array(['mAP50(%)', 'mAP(%)', '参数量(M)', '计算量(GFLOPs)', '模型大小(M)'])

# 每个模型的性能数据
stats = [
    np.array([76.0, 44.5, 17, 43, 38]),
    np.array([73.7, 49.4, 47, 114, 62]),
    np.array([73.8, 42.9, 60, 132, 123]),
    np.array([81.1, 49.2, 21, 58, 50]),
    np.array([70.4, 44.4, 26, 110, 61]),
    np.array([81.4, 54.5, 31, 89, 63]),
    np.array([85.3, 56.2, 30, 85, 61])
]

# 模型名称或编号
model_names = ['YOLOv5n', 'YOLOv6n', 'YOLOv7tiny', 'YOLOXn', 'YOLOv9n', 'YOLOv8n', 'TD-YOLOv8模型']

# 数据归一化
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 绘制每个模型的雷达图
for stat, name in zip(stats, model_names):
    stat = np.concatenate((stat, [stat[0]]))  # 闭合图形
    ax.fill(angles, stat, alpha=0.25)
    ax.plot(angles, stat, label=name)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))

plt.savefig('radar.png', bbox_inches='tight', dpi=1000)

# plt.show()