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

# 3. 计算碰撞风险
def compute_risk(grid_x, grid_y, vehicle_data):
    """
    计算交叉口网格中的碰撞风险。
    grid_x, grid_y: 网格点坐标
    vehicle_data: 车辆轨迹数据，包含质量、速度、位置等信息。
    """
    risk_map = np.zeros((len(grid_x), len(grid_y)))
    
    # 设定参数（可以根据实验调整）
    lambda_1, lambda_2 = 1.0, 0.5  # 加权因子
    beta, alpha = 0.3, 0.2  # 指数衰减因子
    delta_1, delta_2 = 1.0, 1.0  # 距离归一化参数
    S_h = 0.1  # 安全距离
    l, w = 2.0, 1.5  # 车辆标准尺寸
    m_i = 1000  # 自车质量
    m_j = 1000  #目标车辆质量
    
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            risk = 0
            for vehicle in vehicle_data:
                v_i = np.linalg.norm(vehicle["velocity"])  # 车辆速度
                px, py = vehicle["position"]  # 车辆位置
                v_rel = np.linalg.norm(vehicle["velocity"])  # 近似相对速度
                a =  vehicle["a"]
                # 计算 k'
                k_prime = np.sqrt(delta_1 * ((x - px) / (l * np.exp(alpha * v_i)))**2 +
                                  delta_2 * ((y - py) / w)**2)
                
                # 计算距离归一化项
                distance_term = k_prime
                
                # 计算风险函数中的各项
                weight_factor = (1.566 * m_i * v_i**6.687 * 1e-14 + 0.3345) * lambda_1
                speed_factor = (v_rel + S_h) / (distance_term * np.cos(vehicle["phi"]))
                exp_factor = np.exp(lambda_2 * m_j) * np.exp(-beta * a * np.cos(vehicle["theta"]))
                direction_factor = k_prime / abs(k_prime) if k_prime != 0 else 1
                
                # 计算风险值
                risk += weight_factor * (speed_factor**2) * exp_factor * direction_factor
            
            risk_map[i, j] = risk
    
    return risk_map


# 4. 生成热力图
def plot_heatmap(risk_map, grid_x, grid_y):
    plt.figure(figsize=(8, 6))
    sns.heatmap(risk_map.T, xticklabels=grid_x, yticklabels=grid_y, cmap="coolwarm", cbar=True)
    plt.title("Collision Risk Heatmap")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

# 主程序
if __name__ == "__main__":
    map_file = r"C:\Users\LEI\Desktop\intersection_map.json"  # 交叉口地图数据
    vehicle_file = r"C:\Users\LEI\Desktop\vehicle_data.json"  # 车辆轨迹数据

    map_data = load_map_data(map_file)
    vehicle_data = load_vehicle_data(vehicle_file)

    # 定义网格范围（假设交叉口的范围）
    grid_x = np.linspace(-50, 50, 100)  # x 方向
    grid_y = np.linspace(-50, 50, 100)  # y 方向

    risk_map = compute_risk(grid_x, grid_y, vehicle_data)
    plot_heatmap(risk_map, grid_x, grid_y)
