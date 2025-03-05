import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import vectorized
from shapely.prepared import prep

# ---------------------- 辅助函数（道路风险场部分） ---------------------- #
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

# ---------------------- 道路风险场参数与数据处理 ---------------------- #
map_file = r"C:\Users\LEI\Desktop\风险场模型\地图\rotated_map.json"
gamma_1 = 0.5    # 风险基准值
gamma_2 = 0.4    # 风险增长系数
grid_size = 0.1  # 栅格分辨率（米）
plot_range = [1550, 1600, 3945, 3970]  # [min_x, max_x, min_y, max_y]

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

# 生成统一的网格（同时用于道路和车辆风险场计算）
x_grid = np.arange(plot_range[0], plot_range[1], grid_size)
y_grid = np.arange(plot_range[2], plot_range[3], grid_size)
X, Y = np.meshgrid(x_grid, y_grid)

# 初始化道路风险场数组
Z_road = np.full(X.shape, np.nan)
computed_mask = np.zeros(X.shape, dtype=bool)

# ---------------------- 计算道路风险场（矢量化计算） ---------------------- #
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
    Z_road[inside_idx] = risk
    computed_mask[inside_idx] = True  # 标记这些点已经计算

# ---------------------- 车辆风险场部分 ---------------------- #
def translation_transform(x, y, x_i, y_i):
    x_rel = x - x_i
    y_rel = y - y_i
    return x_rel, y_rel

def rotation_transform(x_rel, y_rel, theta_i):
    cos_theta = math.cos(theta_i)
    sin_theta = math.sin(-theta_i)
    x_prime = x_rel * cos_theta - y_rel * sin_theta
    y_prime = x_rel * sin_theta + y_rel * cos_theta
    return x_prime, y_prime

def E_i(x, y, x_i, y_i, beta_x, beta_y, L, W, a_i, v_i, m_i, theta_i, lambda_value, mu, gamma):
    # 计算目标车辆周围任意一点跟质心连线与车辆航向之间的夹角
    phi = np.arctan2(y - y_i, x - x_i) - np.arctan2(math.sin(theta_i), math.cos(theta_i))
    # 计算距离衰减
    delta_x = beta_x * max(abs(x) - 0.5 * L, 0)
    delta_y = beta_y * max(abs(y) - 0.5 * W, 0)
    delta = math.sqrt(delta_x**2 + delta_y**2)
    # 计算速度放大项
    term2 = np.where(np.exp(gamma * v_i * np.cos(phi)) > 1, np.exp(gamma * v_i * np.cos(phi)), 1)
    # 计算加速度质量放大项
    term1 = (1.566 * m_i * (v_i**6.687) * 1e-14 + 0.3345) * lambda_value * max(1, math.exp(mu * a_i * math.cos(phi)))
    return term1 * term2 / (delta + 1)

# 待定系数（车辆风险场）
beta_x = 0.2
beta_y = 0.5
lambda_value = 1.0
mu = 0.05
gamma_param = 0.03  # 为避免与math.gamma冲突，这里用gamma_param

# 车辆参数配置
# 注意：为了使车辆落在道路区域内，这里将车辆的全局坐标调整到道路网格范围内
vehicles = [{
    'x_i': 1580,       # 车辆质心全局坐标x（调整后在道路区域内）
    'y_i': 3954.8,       # 车辆质心全局坐标y
    'theta_i': np.deg2rad(0),   # 航向角
    'L': 4.8,         # 车长
    'W': 1.8,         # 车宽
    'a_i': 0,         # 加速度
    'v_i': 0,        # 速度
    'm_i': 2,         # 质量
}]

# 计算车辆风险场（遍历网格每一点）
Z_vehicle = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        total_risk = 0
        x_global = X[i, j]
        y_global = Y[i, j]
        for vehicle in vehicles:
            # 坐标变换到车辆局部坐标系
            x_rel, y_rel = translation_transform(x_global, y_global, vehicle['x_i'], vehicle['y_i'])
            x_prime, y_prime = rotation_transform(x_rel, y_rel, vehicle['theta_i'])
            # 计算该车辆贡献的风险
            risk = E_i(
                x_prime, y_prime,
                x_i=0, y_i=0,  # 局部坐标系下质心为原点
                beta_x=beta_x,
                beta_y=beta_y,
                L=vehicle['L'],
                W=vehicle['W'],
                a_i=vehicle['a_i'],
                v_i=vehicle['v_i'],
                m_i=vehicle['m_i'],
                theta_i=0,  # 局部坐标系下航向角为0
                lambda_value=lambda_value,
                mu=mu,
                gamma=gamma_param
            )
            total_risk += risk
        Z_vehicle[i, j] = total_risk

# ---------------------- 综合风险场（道路 + 车辆） ---------------------- #
# 若道路风险场某些点未计算（nan），则视为0风险
Z_combined = 0.1*np.nan_to_num(Z_road, nan=0) + 5*Z_vehicle

# ---------------------- 绘制综合风险场热力图 ---------------------- #
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(plot_range[0], plot_range[1])
ax.set_ylim(plot_range[2], plot_range[3])
ax.set_aspect('equal')

contour = ax.contourf(X, Y, Z_combined, levels=50, cmap='jet', antialiased=True)
plt.colorbar(contour, label='Combined Risk Level', ax=ax)

# 叠加车道线
for lane_id, lane_info in lanes.items():
    left = extract_coordinates(lane_info["left_boundary"])
    right = extract_coordinates(lane_info["right_boundary"])
    center = extract_coordinates(lane_info["centerline"])
    
    ax.plot(*zip(*left), color="navy", linewidth=1)
    ax.plot(*zip(*right), color="navy", linewidth=1)
    ax.plot(*zip(*center), color="white", linewidth=1, linestyle="--")

# 叠加车辆本体及其方向箭头
for vehicle in vehicles:
    vehicle_corners = np.array([[-vehicle['L']/2, -vehicle['W']/2],
                                [vehicle['L']/2, -vehicle['W']/2],
                                [vehicle['L']/2, vehicle['W']/2],
                                [-vehicle['L']/2, vehicle['W']/2],
                                [-vehicle['L']/2, -vehicle['W']/2]])
    rot_mat = np.array([[np.cos(vehicle['theta_i']), -np.sin(vehicle['theta_i'])],
                        [np.sin(vehicle['theta_i']), np.cos(vehicle['theta_i'])]])
    rotated_corners = (rot_mat @ vehicle_corners.T).T + [vehicle['x_i'], vehicle['y_i']]
    ax.plot(rotated_corners[:,0], rotated_corners[:,1], linewidth=2, color='black', linestyle='--', label='Vehicle Body')
    
    # 绘制速度方向箭头
    arrow_length_v = 3  # 速度箭头长度
    vx = arrow_length_v * np.cos(vehicle['theta_i'])
    vy = arrow_length_v * np.sin(vehicle['theta_i'])
    ax.arrow(vehicle['x_i'], vehicle['y_i'], vx, vy, color='blue', width=0.1, label='Velocity Direction')
    
    # 绘制加速度方向箭头
    arrow_length_a = 2  # 加速度箭头长度
    ax_a = arrow_length_a * np.cos(vehicle['theta_i'])
    ay_a = arrow_length_a * np.sin(vehicle['theta_i'])
    ax.arrow(vehicle['x_i'], vehicle['y_i'], ax_a, ay_a, color='red', width=0.05, label='Acceleration Direction', linestyle='--')
    
# 处理图例重复问题
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

# 标明车辆参数信息
vehicle = vehicles[0]
speed = vehicle['v_i']
acceleration = vehicle['a_i']
heading_angle = np.rad2deg(vehicle['theta_i'])
text = f'Speed: {speed} m/s, Acceleration: {acceleration} m/s², Heading Angle: {heading_angle:.2f}°'
ax.text(0.5, 1.05, text, transform=ax.transAxes, ha='center', va='bottom')

ax.set_xlabel('World X Coordinate')
ax.set_ylabel('World Y Coordinate')
ax.set_title('Combined Road and Vehicle Risk Field')
plt.tight_layout()
plt.show()
