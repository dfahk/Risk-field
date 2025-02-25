import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 1. 读取交叉口地图数据
def load_map_data(map_file):
    with open(map_file, 'r') as f:
        map_data = json.load(f)
    return map_data  # 这里可以解析车道线、停止线等

# 2. 读取车辆轨迹数据
def load_vehicle_data(vehicle_file):
    with open(vehicle_file, 'r') as f:
        vehicle_data = json.load(f)
    return vehicle_data  # 车辆轨迹信息

# 3. 计算碰撞风险函数
def compute_collision_risk(x, y, x0, y0, vx, vy, ax, ay, mi, vi, vi_self_x, vi_self_y):
    """
    计算坐标 (x, y) 的碰撞风险值，基于公式：
    
    R = (1.566 * mi * vi^6.687 * 1e-14 + 0.3345) * λ1 * 
        [ (|v_rel| + S_h) / sqrt(δ1 * ( (x-x0) / (l*e^(α*v)) )^2 + δ2 * ( (y-y0) / w )^2 ) * cosφ ]^2 
        * e^(λ2*mi) * e^(-βa cosθ) * (k'/|k'|)
    """
    δ1, δ2 = 1.0, 1.0  # 经验参数
    le, w = 2.5, 1.8  # 车辆尺寸
    λ1, λ2 = 1.0, 1.0  # 经验参数
    Sh = 5  # 经验参数
    β = 0.5  # 经验参数
    α = 0.1  # 经验参数

    # 计算相对速度 v_rel（目标车辆速度与自车速度的差）
    v_rel_x = vx - vi_self_x  # 速度差的 x 分量
    v_rel_y = vy - vi_self_y  # 速度差的 y 分量
    v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2)  # 计算相对速度大小

    # 计算相对距离 k'
    k_prime = np.sqrt(δ1 * ((x - x0) / (le * np.exp(α * vi)))**2 + δ2 * ((y - y0) / w)**2)

    # 计算夹角 θ
    theta = np.arctan2(y - y0, x - x0)

    # 计算碰撞风险 R
    risk = (1.566 * mi * vi**6.687 * 1e-14 + 0.3345) * λ1 * (
        ((v_rel + Sh) / k_prime) ** 2 * np.cos(theta)
    ) * np.exp(λ2 * mi) * np.exp(-β * ay * np.cos(theta)) * (k_prime / abs(k_prime))

    return risk

# 计算每个网格点的风险值
Z = np.array([[compute_collision_risk(x, y, x0, y0, vx, vy, ax, ay, mi, vi, vi_self_x, vi_self_y) 
               for x in x_range] for y in y_range])

# 4. 绘制热力图
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=50, cmap='hot', norm=LogNorm())  # 归一化显示风险值
plt.colorbar(label="Collision Risk")
plt.scatter(x0, y0, color='blue', label="Target Vehicle")  # 目标车辆位置
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Collision Risk Heatmap")
plt.legend()
plt.show()

# 主程序
if __name__ == "__main__":
    map_file = "intersection_map.json"  # 交叉口地图数据
    vehicle_file = "vehicle_data.json"  # 车辆轨迹数据

    map_data = load_map_data(map_file)
    vehicle_data = load_vehicle_data(vehicle_file)

    # 定义网格范围（假设交叉口的范围）
    grid_x = np.linspace(-50, 50, 100)  # x 方向
    grid_y = np.linspace(-50, 50, 100)  # y 方向

    risk_map = compute_risk(grid_x, grid_y, vehicle_data)
    plot_heatmap(risk_map, grid_x, grid_y)
