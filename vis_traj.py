import matplotlib
# 设置Matplotlib后端为'TkAgg'，解决PyCharm显示问题
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import numpy as np

# 从文件中加载JSON数据
with open('vehicle_data.json', 'r') as file:
    trajectory_data = json.load(file)

# 提取位置数据和方向数据
x_positions = [point['data']['position'][0] for point in trajectory_data]
y_positions = [point['data']['position'][1] for point in trajectory_data]
directions = [point['data']['direction'] for point in trajectory_data]

# 创建2D轨迹图
plt.figure(figsize=(20, 20))
plt.plot(x_positions, y_positions, marker='o', linestyle='-', color='b', label='Trajectory', markersize=8)


# 标记起点和终点
plt.scatter(x_positions[0], y_positions[0], color='red', s=100, zorder=5, label='Start Point')  # 起点
plt.scatter(x_positions[-1], y_positions[-1], color='green', s=100, zorder=5, label='End Point')  # 终点


fixed_length = 1
# 标记方向，使用归一化和固定长度
# 标记方向，使用归一化和固定长度
for i in range(len(x_positions)):
    # 归一化方向向量
    direction = np.array(directions[i])
    direction_norm = direction / np.linalg.norm(direction)  # 归一化
    # 使用固定长度
    direction_scaled = direction_norm * fixed_length

    # 使用annotate绘制箭头
    plt.annotate('', xy=(x_positions[i] + direction_scaled[0], y_positions[i] + direction_scaled[1]),
                 xytext=(x_positions[i], y_positions[i]),
                 arrowprops=dict(facecolor='red', edgecolor='black', arrowstyle='->', lw=2))
# 设置标题和标签
plt.title('2D Trajectory with Direction')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)

# 显示图例
plt.legend()

# 使用show()显示图形，并确保阻塞主线程直到关闭图形窗口
# plt.show(block=True)
plt.savefig("vehicle_data.png")
