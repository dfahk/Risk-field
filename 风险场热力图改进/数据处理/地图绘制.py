import os
import json
import matplotlib.pyplot as plt

# 设置地图文件路径
map_file = r"C:\Users\LEI\Desktop\风险场模型\地图\rotated_map_cropped.json"

def extract_coordinates(coord_list):
    """从字符串列表中提取坐标元组"""
    return [tuple(map(float, coord.strip("()").split(", "))) for coord in coord_list]

# 读取地图数据
with open(map_file, "r") as f:
    road_data = json.load(f)
lanes = road_data.get("LANE", {})

# 创建画布
fig, ax = plt.subplots(figsize=(12, 10))

# 绘制车道边界和中心线
for lane_id, lane_info in lanes.items():
    # 确保数据中存在边界字段
    if "left_boundary" in lane_info and "right_boundary" in lane_info:
        # 左边界（实线）
        left_boundary = extract_coordinates(lane_info["left_boundary"])
        left_x, left_y = zip(*left_boundary)
        ax.plot(left_x, left_y, color="gray", linewidth=1, linestyle="-")
        
        # 右边界（实线）
        right_boundary = extract_coordinates(lane_info["right_boundary"])
        right_x, right_y = zip(*right_boundary)
        ax.plot(right_x, right_y, color="gray", linewidth=1, linestyle="-")
    
    # 中心线（虚线）
    centerline = extract_coordinates(lane_info["centerline"])
    center_x, center_y = zip(*centerline)
    ax.plot(center_x, center_y, color="gray", linewidth=1, linestyle="--")

# 手动指定显示范围
min_x, max_x = 1550, 1600
min_y, max_y = 3945, 3970

# 设置坐标轴范围和网格
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_aspect("auto")  # 或调整为 "equal" 根据需求
ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5)  # 明确启用并设置网格样式
ax.set_xlabel("X (World Coordinates)")
ax.set_ylabel("Y (World Coordinates)")
ax.set_title("Road Map with Lane Boundaries and Grid")

plt.tight_layout()
plt.show()