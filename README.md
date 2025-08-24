collect_traj.py
运行车辆，实时采集车辆状态（位置、速度、方向等），并保存为 vehicle_data.json，作为后续轨迹跟踪的参考。

下面两个有一个肯定能运行：

controller.py
读取 vehicle_data.json，使用 MPC（模型预测控制）进行轨迹跟踪，动态调整车辆转向、油门和刹车，保证车辆沿预设轨迹行驶。

italy_controller.py
针对 BeamNG 自带 italy 场景的测试控制器。演示在复杂场景下运行 MPC 控制器。
