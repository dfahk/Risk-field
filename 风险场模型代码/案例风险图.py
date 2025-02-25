import numpy as np
import matplotlib.pyplot  as plt
# ========== 输入变量（示例值）==========
xi = 15.0    # 自车X坐标
yi = 5.0    # 自车Y坐标
xe = 0   # 目标车X坐标
ye = 0    # 目标车Y坐标
v_e = 10.0  # 自车速度(m/s)
v_i = 8.0   # 目标车速度(m/s)
varphi_e = np.deg2rad(30)   # 自车航向角
varphi_i = np.deg2rad(45)   # 目标车航向角
a_i = 6#目标车辆加速度

#属性变量
m_i = 3
l = 1.0
w = 1.0

#待定系数
alpha = 1.0
beta = 1.0
lambda_value = 1.0
theta = 1.0
delta_1 = 1.0
delta_2 = 1.0
gamma = 1.0
mu = 1.0 

# 定义变量
vx_rel = (v_e * np.cos(varphi_e)) - (v_i * np.cos(varphi_i)) 
vy_rel = (v_e * np.sin(varphi_e)) - (v_i * np.sin(varphi_i)) 
px_rel = xe - xi
py_rel = ye - yi
v_rel = np.array([vx_rel, vy_rel])  # 相对速度向量 (vx_rel, vy_rel)
P_rel = np.array([px_rel, py_rel])  # 相对位置向量 (px_rel, py_rel)

# 计算 E_i(x, y) 的函数
def E_i(x, y, xi, yi, m_i, v_i, a_i, l, w, alpha, beta, lambda_value, theta, delta_1, delta_2, gamma, mu):
    # 计算公式中的各项
    term1 = (1.566 * m_i * (v_i ** 6.687) * 10**-14 + 0.3345) * lambda_value * np.exp(-mu * a_i * np.cos(theta))
    term2 = np.sqrt(delta_1 * ((x - xi) / (l * np.exp(gamma * v_i))) ** 2 + delta_2 * ((y - yi) / w) ** 2)
    term3 = alpha * np.exp(   # 修正写法 
    (-beta * np.linalg.norm(v_rel)**2)  / (np.dot(P_rel,  v_rel))
)
    # 最终的公式结果
    result = term3  * term1 / term2
    return result

# ========== 参数配置 ==========
x_range = (xi-20, xi+20)  # X轴显示范围
y_range = (yi-10, yi+20)  # Y轴显示范围
grid_resolution = 0.5      # 网格分辨率（值越小精度越高）

# ========== 生成网格 ==========
x = np.linspace(x_range[0],  x_range[1], int((x_range[1]-x_range[0])/grid_resolution))
y = np.linspace(y_range[0],  y_range[1], int((y_range[1]-y_range[0])/grid_resolution))
X, Y = np.meshgrid(x,  y)

# ========== 计算风险场 ==========
Z = E_i(X, Y, xi, yi, m_i, v_i, a_i, l, w, 
        alpha, beta, lambda_value, theta, 
        delta_1, delta_2, gamma, mu)

# ========== 可视化 ==========
plt.figure(figsize=(12,  8), dpi=100)
heatmap = plt.pcolormesh(X,  Y, Z, shading='auto', cmap='jet_r', alpha=0.9)
plt.colorbar(heatmap,  label='风险等级', extend='max')

# 标记车辆位置
plt.scatter(xi,  yi, s=200, c='lime', marker='s', edgecolor='black', label='自车')
plt.scatter(xe,  ye, s=200, c='red', marker='^', edgecolor='black', label='目标车')

# 增强可视化效果
plt.title(' 车辆运动风险场热力图\n(高亮区域表示碰撞风险)', pad=20, fontsize=14)
plt.xlabel('X 坐标 (m)', labelpad=10)
plt.ylabel('Y 坐标 (m)', labelpad=10)
plt.grid(True,  alpha=0.3)
plt.legend(loc='upper  right', framealpha=0.9)  
plt.tight_layout() 
plt.show() 