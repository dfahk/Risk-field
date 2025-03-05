import os
import json
import matplotlib.pyplot as plt

# 设置地图文件路径
map_file = r"C:\Users\LEI\Desktop\风险场模型\地图\rotated_map.json"

def extract_coordinates(coord_list):
    """从字符串列表中提取坐标元组"""
    return [tuple(map(float, coord.strip("()").split(", "))) for coord in coord_list]

# 读取地图数据
with open(map_file, "r") as f:
    road_data = json.load(f)
lanes = road_data.get("LANE", {})

# 定义显示范围
min_x, max_x = 1550, 1600
min_y, max_y = 3945, 3970

# 筛选在显示范围内的车道数据
filtered_lanes = {}
for lane_id, lane_info in lanes.items():
    all_points = []
    # 提取所有相关点（左/右边界、中心线）
    for key in ["left_boundary", "right_boundary", "centerline"]:
        if key in lane_info:
            all_points.extend(extract_coordinates(lane_info[key]))
    # 检查是否至少有一个点在范围内
    keep_lane = False
    for x, y in all_points:
        if (min_x <= x <= max_x) and (min_y <= y <= max_y):
            keep_lane = True
            break
    if keep_lane:
        filtered_lanes[lane_id] = lane_info

# 创建新地图数据并保存
new_road_data = road_data.copy()
new_road_data["LANE"] = filtered_lanes
new_file = os.path.splitext(map_file)[0] + "_cropped.json"
with open(new_file, "w") as f:
    json.dump(new_road_data, f, indent=4)
print(f"已保存截取数据至: {new_file}")

# 创建画布并绘制原始数据（原文件不变）
fig, ax = plt.subplots(figsize=(12, 10))
for lane_id, lane_info in lanes.items():
    if "left_boundary" in lane_info and "right_boundary" in lane_info:
        # 绘制左边界
        left_boundary = extract_coordinates(lane_info["left_boundary"])
        left_x, left_y = zip(*left_boundary)
        ax.plot(left_x, left_y, color="gray", linewidth=1, linestyle="-")
        # 绘制右边界
        right_boundary = extract_coordinates(lane_info["right_boundary"])
        right_x, right_y = zip(*right_boundary)
        ax.plot(right_x, right_y, color="gray", linewidth=1, linestyle="-")
    # 绘制中心线
    centerline = extract_coordinates(lane_info["centerline"])
    center_x, center_y = zip(*centerline)
    ax.plot(center_x, center_y, color="gray", linewidth=1, linestyle="--")

# 设置显示范围
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_aspect("auto")
ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5)
ax.set_xlabel("X (World Coordinates)")
ax.set_ylabel("Y (World Coordinates)")
ax.set_title("Road Map with Lane Boundaries and Grid")

plt.tight_layout()
plt.show()