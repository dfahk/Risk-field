import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString

def extract_coordinates(coord_list):
    """从字符串列表中提取坐标元组"""
    return [tuple(map(float, coord.strip("()").split(", "))) for coord in coord_list]

# 参数配置
map_file = r"C:\Users\LEI\Desktop\风险场模型\地图\rotated_map.json"
gamma_1 = 1.0    # 风险基准值
gamma_2 = 0.1    # 风险增长系数
grid_size = 0.1   # 栅格分辨率（米）
plot_range = [1550, 1600, 3945, 3970]  # [min_x, max_x, min_y, max_y]

# 读取预旋转的地图数据
with open(map_file, "r") as f:
    road_data = json.load(f)
lanes = road_data.get("LANE", {})

# 初始化图形
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(plot_range[:2])
ax.set_ylim(plot_range[2:])
ax.set_aspect('equal')

# 预处理车道几何数据
lane_polygons = []
for lane_id, lane_info in lanes.items():
    # 解析坐标数据
    left = extract_coordinates(lane_info["left_boundary"])
    right = extract_coordinates(lane_info["right_boundary"])
    center = extract_coordinates(lane_info["centerline"])
    
    # 构建几何对象
    polygon = Polygon(left + right[::-1])  # 注意右边界需要反向连接
    lane_polygons.append(polygon)
    
    # 存储几何数据
    lane_info['geometry'] = {
        'polygon': polygon,
        'left_segments': [LineString([left[i], left[i+1]]) for i in range(len(left)-1)],
        'right_segments': [LineString([right[i], right[i+1]]) for i in range(len(right)-1)],
        'centerline': center
    }

# 生成网格坐标系
x_grid = np.arange(plot_range[0], plot_range[1], grid_size)
y_grid = np.arange(plot_range[2], plot_range[3], grid_size)
X, Y = np.meshgrid(x_grid, y_grid)
Z = np.full_like(X, np.nan)

# 风险场计算
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = Point(X[i,j], Y[i,j])
        for lane_id, lane_info in lanes.items():
            geom = lane_info['geometry']
            if geom['polygon'].contains(point):
                # 计算到边界的距离
                d_left = min(seg.distance(point) for seg in geom['left_segments'])
                d_right = min(seg.distance(point) for seg in geom['right_segments'])
                Z[i,j] = gamma_1 * math.exp(gamma_2 * abs(d_left - d_right))
                break

# 创建等高线热力图
contour = ax.contourf(
    X, Y, Z, 
    levels=50,
    cmap='jet',
    antialiased=True
)
plt.colorbar(contour, label='Risk Level', ax=ax)

# 叠加车道线
for lane_id, lane_info in lanes.items():
    left = extract_coordinates(lane_info["left_boundary"])
    right = extract_coordinates(lane_info["right_boundary"])
    center = extract_coordinates(lane_info["centerline"])
    
    # 绘制边界
    ax.plot(*zip(*left), color="navy", linewidth=1, linestyle="-")
    ax.plot(*zip(*right), color="navy", linewidth=1, linestyle="-")
    
    # 绘制中心线
    ax.plot(*zip(*center), color="white", linewidth=1, linestyle="--")

# 图形美化
ax.grid(False)
ax.set_xlabel('World X Coordinate')
ax.set_ylabel('World Y Coordinate')
ax.set_title('Road Risk Heatmap Visualization')
plt.tight_layout()
plt.show()