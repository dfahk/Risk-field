import numpy as np
import matplotlib.pyplot  as plt 
from matplotlib.colors  import LogNorm
plt.rcParams['font.sans-serif']  = ['Microsoft YaHei']  # 微软雅黑字体 
plt.rcParams['axes.unicode_minus']  = False  # 解决负号显示异常 
# 输入变量（请根据实际情况填入值）
xi = 5  # 目标车x坐标
yi = 5  # 目标车y坐标
xe = 0  # 自车x坐标
ye = 0  # 自车y坐标
v_e = 0  # 自车速度
v_i = 0  # 目标车速度
varphi_e = -np.pi  # 自车偏航角
varphi_i = -np.pi/6  # 目标车偏航角
a_i = 0 # 目标车辆加速度

# 属性变量
m_i = 3
l = 4
w = 1.5

# 待定系数
alpha = 1
beta = 1
lambda_value = 1.0
delta_1 = 16
delta_2 = 2.25
gamma_1 = 0.05  
gamma_2 = 0.05
mu = 0.1 

# 定义变量
vx_rel = (v_e * np.cos(varphi_e)) - (v_i * np.cos(varphi_i)) 
vy_rel = (v_e * np.sin(varphi_e)) - (v_i * np.sin(varphi_i)) 
px_rel = xe - xi
py_rel = ye - yi
v_rel = np.array([vx_rel, vy_rel])  # 相对速度向量 (vx_rel, vy_rel)
P_rel = np.array([px_rel, py_rel])  # 相对位置向量 (px_rel, py_rel)

# 计算 E_i(x, y) 的函数
def E_i(x, y, xi, yi, m_i, v_i, a_i, l, w, alpha, beta, lambda_value, delta_1, delta_2, gamma_1, gamma_2, mu):
    
    # 计算目标车辆的速度方向与连线之间的夹角 eta
    eta = np.arctan2(y - yi, x - xi) - np.arctan2(np.sin(varphi_i), np.cos(varphi_i))
    theta = eta
    # 计算公式中的各项
    term1 = (1.566 * m_i * (v_i ** 6.687) * 10**-14 + 0.3345) * lambda_value * np.exp(mu * a_i * np.cos(theta))
    term2 = np.sqrt(delta_1 * ((x - xi) / (l * np.exp(gamma_1 * v_i*np.cos(eta)))) ** 2 + delta_2 * ((y - yi) / (w * np.exp(gamma_2 * v_i*np.cos(eta)))) ** 2)
    term3 = 1  # 修改这里
    # 最终的公式结果
    result = term3 * term1 / term2
    return result
#------------------------ 绘图代码 ------------------------#
# 生成网格数据（范围覆盖自车和目标车坐标）
x_range = np.linspace(-5,  15, 300)  # x轴范围 
y_range = np.linspace(-5,  15, 300)  # y轴范围 
X, Y = np.meshgrid(x_range,  y_range)
 
# 向量化计算函数（避免修改原函数）
calc_risk = np.vectorize(E_i,  excluded=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
 
# 计算整个网格的风险值矩阵 
Z = calc_risk(X, Y, 
              xi, yi,        # 自车坐标 
              m_i, v_i, a_i, # 目标车属性 
              l, w,          # 车辆尺寸 
              alpha, beta, lambda_value, delta_1, delta_2, gamma_1, gamma_2, mu) # 待定系数 
 
# 创建带标注的热力图 
plt.figure(figsize=(12,  9))
Z_nonzero = np.where(Z  <= 0, 1e-6, Z)  # 避免零值报错
heatmap = plt.pcolormesh(X,  Y, Z_nonzero, shading='auto', cmap='jet', norm=LogNorm())
 
# 添加要素标注 
plt.colorbar(heatmap,  label='Risk Value', extend='max')
plt.scatter(xi,  yi, c='lime', s=120, edgecolors='black', label='目标车位置')
plt.scatter(xe,  ye, c='red', s=120, marker='X', label='自车位置')
plt.quiver(xe,  ye, 
           v_e*np.cos(varphi_e),  v_e*np.sin(varphi_e),  
           color='lime', scale=40, width=0.002, label='自车速度方向')
plt.quiver(xi,  yi, 
           v_i*np.cos(varphi_i),  v_i*np.sin(varphi_i),  
           color='red', scale=40, width=0.002, label='目标车速度方向')
 
# 设置图表参数 
plt.title(' 车辆交互风险场可视化\n'
         f'自车速度：{v_e}m/s | 目标车速度：{v_i}m/s | 目标车加速度：{a_i}m/s²', 
         fontsize=14, pad=20)
plt.xlabel('X 坐标 (m)', fontsize=12)
plt.ylabel('Y 坐标 (m)', fontsize=12)
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(ls='--',  alpha=0.5)
plt.axis('equal') 
plt.tight_layout() 
plt.show() 