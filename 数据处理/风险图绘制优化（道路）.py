import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import vectorized
from shapely.prepared import prep

def extract_coordinates(coord_list):
    """从字符串列表中提取坐标元组"""
    return [tuple(map(float, coord.strip("()").split(", "))) for coord in coord_list]

def point_to_segment_distance(x, y, x1, y1, x2, y2):
    """
    计算点(x,y)到线段[(x1,y1),(x2,y2)]的距离，支持向量化计算
    """
    dx = x2 - x1
    dy = y2 - y1
    denom = dx**2 + dy**2
    # 避免除零错误（若线段退化为点）
    t = ((x - x1) * dx + (y - y1) * dy) / (denom if denom != 0 else 1)
    t_clamped = np.clip(t, 0, 1)
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    return np.sqrt((x - proj_x)**2 + (y - proj_y)**2)

# ---------------------- 参数配置 ---------------------- #
map_file = r"C:\Users\LEI\Desktop\风险场模型\地图\rotated_map.json"
gamma_1 = 1.0    # 风险基准值
gamma_2 = 0.1    # 风险增长系数
grid_size = 0.1  # 栅格分辨率（米）
plot_range = [1550, 1600, 3945, 3970]  # [min_x, max_x, min_y, max_y]

# ---------------------- 读取并预处理地图数据 ---------------------- #
with open(map_file, "r") as f:
    road_data = json.load(f)
lanes = road_data.get("LANE", {})

for lane_id, lane_info in lanes.items():
    # 解析左右边界与中心线
    left = extract_coordinates(lane_info["left_boundary"])
    right = extract_coordinates(lane_info["right_boundary"])
    center = extract_coordinates(lane_info["centerline"])
    
    # 构建车道多边形（右边界逆序拼接）
    polygon = Polygon(left + right[::-1])
    prepared_polygon = prep(polygon)  # 预处理几何对象，加速 contains 测试
    
    # 存储几何数据与线段（直接保存端点对，用于后续距离计算）
    lane_info['geometry'] = {
        'polygon': polygon,
        'prepared_polygon': prepared_polygon,
        'left_segments': [(left[i], left[i+1]) for i in range(len(left)-1)],
        'right_segments': [(right[i], right[i+1]) for i in range(len(right)-1)],
        'centerline': center
    }

# ---------------------- 生成网格坐标 ---------------------- #
x_grid = np.arange(plot_range[0], plot_range[1], grid_size)
y_grid = np.arange(plot_range[2], plot_range[3], grid_size)
X, Y = np.meshgrid(x_grid, y_grid)
Z = np.full(X.shape, np.nan)
# 用于标记哪些栅格点已经计算过风险值（避免重复计算）
computed_mask = np.zeros(X.shape, dtype=bool)

# ---------------------- 风险场计算（矢量化方式） ---------------------- #
for lane_id, lane_info in lanes.items():
    geom = lane_info['geometry']
    polygon = geom['polygon']
    prepared_polygon = geom['prepared_polygon']
    
    # 利用车道多边形的包围盒进行初步过滤，并排除已计算过的点
    minx, miny, maxx, maxy = polygon.bounds
    bbox_mask = (X >= minx) & (X <= maxx) & (Y >= miny) & (Y <= maxy) & (~computed_mask)
    if not np.any(bbox_mask):
        continue

    # 对包围盒内的点利用矢量化 contains 测试判断是否在多边形内
    xs = X[bbox_mask]
    ys = Y[bbox_mask]
    inside = vectorized.contains(polygon, xs, ys)
    if not np.any(inside):
        continue

    # 获取处于车道内部的网格点在整体数组中的索引
    grid_idx = np.where(bbox_mask)
    inside_idx = (grid_idx[0][inside], grid_idx[1][inside])
    xs_inside = X[inside_idx]
    ys_inside = Y[inside_idx]
    
    # 计算每个网格点到左边界各线段的最小距离
    d_left = np.full(xs_inside.shape, np.inf)
    for seg in geom['left_segments']:
        (x1, y1), (x2, y2) = seg
        d = point_to_segment_distance(xs_inside, ys_inside, x1, y1, x2, y2)
        d_left = np.minimum(d_left, d)
    
    # 计算每个网格点到右边界各线段的最小距离
    d_right = np.full(xs_inside.shape, np.inf)
    for seg in geom['right_segments']:
        (x1, y1), (x2, y2) = seg
        d = point_to_segment_distance(xs_inside, ys_inside, x1, y1, x2, y2)
        d_right = np.minimum(d_right, d)
    
    # 计算风险值，并更新风险场数组
    risk = gamma_1 * np.exp(gamma_2 * np.abs(d_left - d_right))
    Z[inside_idx] = risk
    computed_mask[inside_idx] = True  # 标记这些点已经计算

# ---------------------- 绘制风险场热力图 ---------------------- #
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(plot_range[0], plot_range[1])
ax.set_ylim(plot_range[2], plot_range[3])
ax.set_aspect('equal')

contour = ax.contourf(X, Y, Z, levels=50, cmap='jet', antialiased=True)
plt.colorbar(contour, label='Risk Level', ax=ax)

# ---------------------- 叠加车道线 ---------------------- #
for lane_id, lane_info in lanes.items():
    left = extract_coordinates(lane_info["left_boundary"])
    right = extract_coordinates(lane_info["right_boundary"])
    center = extract_coordinates(lane_info["centerline"])
    
    ax.plot(*zip(*left), color="navy", linewidth=1)
    ax.plot(*zip(*right), color="navy", linewidth=1)
    ax.plot(*zip(*center), color="white", linewidth=1, linestyle="--")

ax.set_xlabel('World X Coordinate')
ax.set_ylabel('World Y Coordinate')
ax.set_title('Road Risk Heatmap Visualization')
plt.tight_layout()
plt.show()
