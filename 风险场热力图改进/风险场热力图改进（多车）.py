import numpy as np
import matplotlib.pyplot as plt
import math

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

# 待定系数
beta_x = 0.2
beta_y = 0.5
lambda_value = 1.0
mu = 0.05
gamma = 0.03

# 车辆参数配置，添加第二辆车的参数
vehicles = [
    {
        'x_i': -10,       # 车辆质心全局坐标x
        'y_i': -10,       # 车辆质心全局坐标y
        'theta_i': np.deg2rad(30),   # 航向角
        'L': 4.8,         # 车长
        'W': 1.8,         # 车宽
        'a_i': 3,     # 加速度
        'v_i': 10,      # 速度
        'm_i': 2,    # 质量
    },
    {
        'x_i': -5,      # 第二辆车质心全局坐标x
        'y_i': -5,       # 第二辆车质心全局坐标y
        'theta_i': np.deg2rad(60),   # 第二辆车航向角
        'L': 4.8,         # 车长
        'W': 1.8,         # 车宽
        'a_i': 2,     # 加速度
        'v_i': 8,      # 速度
        'm_i': 1.8,    # 质量
    }
]

# 生成网格
x = np.linspace(-30, 30, 100)
y = np.linspace(-30, 30, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# 计算每个点的风险值
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
                gamma=gamma
            )
            total_risk += risk
        Z[i, j] = total_risk

# 绘制风险场
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=50, cmap='jet')
plt.colorbar(contour, label='Risk Level')
plt.xlabel('Global X (m)')
plt.ylabel('Global Y (m)')
plt.title('Vehicle Risk Field')
plt.grid(True, alpha=0.3)

# 用于存储要显示在图例中的元素
legend_elements = []

# 绘制车辆本体轮廓
for idx, vehicle in enumerate(vehicles):
    vehicle_corners = np.array([[-vehicle['L']/2,  -vehicle['W']/2],
                                [vehicle['L']/2, -vehicle['W']/2],
                                [vehicle['L']/2, vehicle['W']/2],
                                [-vehicle['L']/2, vehicle['W']/2],
                                [-vehicle['L']/2, -vehicle['W']/2]])

    # 对车辆轮廓进行旋转变换
    rot_mat = np.array([[np.cos(vehicle['theta_i']),  -np.sin(vehicle['theta_i'])], 
                        [np.sin(vehicle['theta_i']),  np.cos(vehicle['theta_i'])]]) 
    rotated_corners = (rot_mat @ vehicle_corners.T).T + [vehicle['x_i'], vehicle['y_i']]

    body_line, = plt.plot(rotated_corners[:, 0], rotated_corners[:, 1],
                          linewidth=2, color='black', linestyle='--')
    if idx == 0:
        legend_elements.append((body_line, 'Vehicle Body'))

    # 绘制速度方向箭头
    arrow_length_v = 3  # 速度箭头长度
    vx = arrow_length_v * np.cos(vehicle['theta_i'])
    vy = arrow_length_v * np.sin(vehicle['theta_i'])
    if idx == 0:
        vel_arrow = plt.arrow(vehicle['x_i'], vehicle['y_i'], vx, vy, color='blue', width=0.1)
        legend_elements.append((vel_arrow, 'Velocity Direction'))
    else:
        plt.arrow(vehicle['x_i'], vehicle['y_i'], vx, vy, color='blue', width=0.1)

    # 绘制加速度方向箭头
    arrow_length_a = 2  # 加速度箭头长度
    ax = arrow_length_a * np.cos(vehicle['theta_i'])
    ay = arrow_length_a * np.sin(vehicle['theta_i'])
    if idx == 0:
        # 设置 linestyle='--' 来绘制虚线箭头
        acc_arrow = plt.arrow(vehicle['x_i'], vehicle['y_i'], ax, ay, color='red', width=0.05, linestyle='--')
        legend_elements.append((acc_arrow, 'Acceleration Direction'))
    else:
        plt.arrow(vehicle['x_i'], vehicle['y_i'], ax, ay, color='red', width=0.05, linestyle='--')

# 创建图例
handles = [element[0] for element in legend_elements]
labels = [element[1] for element in legend_elements]
plt.legend(handles, labels)

# 标明速度、加速度和航向角大小
for idx, vehicle in enumerate(vehicles):
    speed = vehicle['v_i']
    acceleration = vehicle['a_i']
    heading_angle = np.rad2deg(vehicle['theta_i'])
    text = f'Vehicle {idx + 1}: Speed: {speed} m/s, Acceleration: {acceleration} m/s², Heading Angle: {heading_angle:.2f}°'
    plt.text(0.5, 1.1 - idx * 0.05, text, transform=plt.gca().transAxes, ha='center', va='bottom')

plt.show()