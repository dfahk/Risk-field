import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# 参数设置
gamma1 = 18  # 可根据实际需求修改
gamma2 = 1   # 可根据实际需求修改
w = 3        # 周期宽度
k = 3        # k 的范围，调整观察范围

# 定义分段函数
def piecewise_function(x, w, k, gamma1, gamma2):
    if w / 2 <= x <= (2 * k - 1) / 2 * w:
        return gamma1 * np.cos(2 * np.pi * x / w) + gamma1
    elif 0 < x < w / 2:
        return gamma2 / x - 2 * gamma2 / w
    elif (2 * k - 1) / 2 * w < x < k * w:
        return -gamma2 / (x - k * w) - 2 * gamma2 / w
    else:
        return 0  # 默认值，函数外部无定义

# 生成数据
resolution = 2000  # 提高分辨率
x_vals = np.linspace(0, k * w, resolution)
y_vals = np.linspace(0, k * w, resolution)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

# 计算 Z 值
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = piecewise_function(X[i, j], w, k, gamma1, gamma2)

# 对 Z 归一化
Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

# 定义自定义颜色映射
colors = [
    (0, 'navy'),
    (0.3, 'royalblue'),
    (0.4, 'cyan'),
    (0.5, 'lightgreen'),
    (0.6, 'yellow'),
    (0.8, 'orange'),
    (1, 'firebrick')
]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# 绘制 3D 图像
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 使用归一化后的 Z 绘制表面图
surface = ax.plot_surface(X, Y, Z_normalized, cmap=cmap, edgecolor='none')
ax.set_title('3D Plot of the Piecewise Function', fontsize=14)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z (Normalized)', fontsize=12)

# 添加颜色条
cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Normalized Function Value', fontsize=12)

plt.show()