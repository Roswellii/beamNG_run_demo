import numpy as np
import time
import json
from beamngpy import BeamNGpy, Scenario, Vehicle
from scipy.optimize import minimize
import random
from beamngpy.sensors import Electrics
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
# 初始化日志配置
from beamngpy import set_up_simple_logging


# 设置随机种子
random.seed(1703)



set_up_simple_logging()


class BeamNGConnection:
    def __init__(self, host="localhost", port=25252):
        self.host = host
        self.port = port
        self.beamng = None
        self.scenario = None
        self.vehicle = None

    def open_connection(self):
        self.beamng = BeamNGpy(self.host, self.port, home=r'C:\Program Files (x86)\BeamNG.tech.v0.34.2.0')
        self.beamng.open()

    def setup_scenario(self, scenario_name="basicScenario", vehicle_name="vehicle"):
        if self.beamng is None:
            raise Exception("BeamNG connection is not open.")

        # 仅在这里初始化并设置场景
        scenario = Scenario(scenario_name, self.beamng)
        vehicle = Vehicle(vehicle_name, model="etk800", licence="ZJU")
        scenario.add_vehicle(
            vehicle, pos=(245.11, -906.94, 247.46), rot_quat=(0.0010, 0.1242, 0.9884, -0.0872)
        )
        scenario.make(self.beamng)  # 创建场景文件
        self.scenario = scenario
        self.vehicle = vehicle
        return scenario, vehicle

    def get_vehicle_data(self):
        if self.vehicle is None:
            raise Exception("Vehicle is not initialized.")

        vehicle_data = self.vehicle.state
        speed = vehicle_data['vel'][0]  # 车速（m/s）
        position = vehicle_data['pos']  # 车辆位姿
        return speed, position

    def start_scenario(self):
        if self.scenario is None:
            raise Exception("Scenario is not initialized.")

        self.beamng.settings.set_deterministic(60)
        self.scenario.load(self.scenario)  # 加载场景
        self.beamng.ui.hide_hud()  # 隐藏HUD
        self.scenario.start()  # 启动场景


class TrajectoryLoader:
    def __init__(self, trajectory_file):
        self.trajectory_file = trajectory_file
        self.trajectory = self.load_trajectory()

    def load_trajectory(self):
        # 读取并解析轨迹文件
        with open(self.trajectory_file, 'r') as file:
            return json.load(file)  # 返回解析后的轨迹列表

    def get_next_point(self, current_position):
        """
        返回离当前车辆位置最近的轨迹点，并返回其目标 (x, y, theta)。
        假设当前车辆位置包含 x, y 坐标。
        """
        closest_point = None
        min_distance = float('inf')

        # 遍历轨迹点，找到离车辆当前位置最近的点
        for point in self.trajectory:
            target_x, target_y, target_theta = point
            distance = np.sqrt((current_position[0] - target_x) ** 2 + (current_position[1] - target_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_point = point

        # 返回最接近的轨迹点
        return closest_point


class MPCController:
    def __init__(self, prediction_horizon=10, target_speed=20):
        self.prediction_horizon = prediction_horizon  # 预测步长
        self.target_speed = target_speed  # 目标车速

    def cost_function(self, control_inputs, current_state, trajectory):
        """
        目标函数：最小化误差（包括位置误差、方向误差和速度误差）
        """
        total_cost = 0
        x, y, theta = current_state  # 当前状态
        for i in range(self.prediction_horizon):
            speed, steering_angle = control_inputs[i * 2:i * 2 + 2]
            target_x, target_y, target_theta = trajectory[i]
            # 计算位置误差
            position_error = np.sqrt((target_x - x) ** 2 + (target_y - y) ** 2)
            # 计算方向误差
            angle_error = np.arctan2(target_y - y, target_x - x) - theta
            # 速度误差
            speed_error = speed - self.target_speed
            # 计算每一步的代价，权重可以调整
            total_cost += position_error + 0.5 * abs(angle_error) + 0.1 * abs(speed_error)
            x += speed * np.cos(theta)  # 更新位置
            y += speed * np.sin(theta)
            theta += steering_angle  # 更新方向
        return total_cost

    def get_control(self, current_state, trajectory):
        """
        使用MPC来优化控制输入
        """
        initial_guess = np.zeros(self.prediction_horizon * 2)  # 假设所有控制输入都为0
        result = minimize(self.cost_function, initial_guess, args=(current_state, trajectory), method='SLSQP')
        return result.x


class VehicleController:
    def __init__(self, beamng_connection, vehicle, traj, mpc_controller):
        self.beamng_connection = beamng_connection
        self.traj = traj
        self.mpc_controller = mpc_controller
        self.previous_time = time.time()
        self.vehicle= vehicle

        # 初始化可视化
        self.fig, self.ax = plt.subplots()
        self.line_traj, = self.ax.plot([], [], 'b-', label="Trajectory")  # 轨迹
        self.scatter_current_pose, = self.ax.plot([], [], 'bo', label="Current Position")  # 当前位置
        self.scatter_current_way_point, = self.ax.plot([], [], 'go', label="Current Way Point")  # 当前位置
        self.scatter_next_way_point, = self.ax.plot([], [], 'ro', label="Next Way Point")  # 下一个轨迹点
        self.scatter_look_ahead, = self.ax.plot([], [], 'yo', label="Look Ahead")  # 前瞄点

        # 设置图形的范围和标签

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Vehicle Tracking and Next Point Visualization')
        self.ax.legend()

        self.ax.set_xlim([-100, 100])  # 设置X轴范围，例如 xmin = -10, xmax = 10
        self.ax.set_ylim([-100, 100])

    def control(steering: float | None = None, throttle: float | None = None, brake: float | None = None,
                parkingbrake: float | None = None, clutch: float | None = None, gear: int | None = None):
        # 这里我们只是打印出控制信号，实际中会调用车辆控制接口
        print(f"Steering: {steering}, Throttle: {throttle}, Brake: {brake}, Gear: {gear}")



    def get_closest_traj_point(self, vehicle_pos):
        """
        提取距离车辆当前位置最近的轨迹点
        """
        closest_point = None
        min_distance = float('inf')  # 初始化为无限大

        for entry in self.traj:
            # 轨迹点的 'position' 是一个包含 [x, y, z] 的列表
            traj_pos = entry["data"]["position"]

            # 计算车辆与轨迹点的欧氏距离，只计算 x 和 y 坐标
            distance = math.sqrt((vehicle_pos[0] - traj_pos[0]) ** 2 + (vehicle_pos[1] - traj_pos[1]) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_point = traj_pos  # 保存最近轨迹点的 position

        return closest_point

    def get_next_traj_point(self, current_point):
        """
        提取当前轨迹点之后的下一个轨迹点
        """
        if current_point is None:
            return None

        # 获取当前轨迹点的索引
        current_index = None
        for i, entry in enumerate(self.traj):
            if entry["data"]["position"] == current_point:
                current_index = i
                break

        # 如果当前轨迹点是最后一个点，返回该点（无下一个轨迹点）
        if current_index is not None and current_index + 1 < len(self.traj):
            return self.traj[current_index + 20]["data"]["position"]
        else:
            return current_point  # 或者返回 None，视需求而定

    def distance(self,p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def calculate_steering_angle(self, vehicle_position, vehicle_orientation, lookahead_point):
        dx = lookahead_point[0] - vehicle_position[0]
        dy = lookahead_point[1] - vehicle_position[1]

        # 目标点和车辆的相对角度
        angle_to_target = math.atan2(dy, dx)

        # If vehicle_orientation is a vector, compute its orientation angle (yaw) using atan2
        vehicle_yaw = math.atan2(vehicle_orientation[1], vehicle_orientation[0])  # Assuming 2D orientation in X-Y plane

        # Calculate the difference in angle
        delta_angle = angle_to_target - vehicle_yaw

        # 保持角度在[-pi, pi]范围内
        if delta_angle > math.pi:
            delta_angle -= 2 * math.pi
        elif delta_angle < -math.pi:
            delta_angle += 2 * math.pi

        return delta_angle

    def calculate_lookahead_point(self, pos, lookahead_distance, next_way_point):
        """
        计算2D前视点，不考虑车辆当前方向
        :param pos: 当前位置 [x, y]
        :param lookahead_distance: 前视距离
        :param next_way_point: 下一个轨迹点 {"timestamp": float, "data": {"position": [x, y, z], ...}}
        :return: 前视点坐标 [x, y]
        """
        # 提取当前位置坐标 (2D)
        curr_x, curr_y = pos[:2]

        # 提取下一个轨迹点的坐标 (2D)
        next_x = next_way_point[0]
        next_y = next_way_point[1]

        # 计算当前位置到下一个轨迹点的向量
        dx = next_x - curr_x
        dy = next_y - curr_y

        # 计算当前位置到下一个轨迹点的距离
        dist_to_next = math.sqrt(dx * dx + dy * dy)

        if dist_to_next == 0:
            # 如果距离为0，直接返回下一个轨迹点
            return [next_x, next_y]

        # 如果距离下一个点已经小于前视距离，直接返回下一个点
        if dist_to_next <= lookahead_distance:
            return [next_x, next_y]

        # 计算前视点
        # 使用当前位置到下一个点的方向，延伸到前视距离
        lookahead_x = curr_x + dx * lookahead_distance / dist_to_next
        lookahead_y = curr_y + dy * lookahead_distance / dist_to_next

        return [lookahead_x, lookahead_y]

    def update(self):
        """
        更新车辆状态和可视化
        """
        self.vehicle.sensors.poll()  # 获取传感器数据
        current_time = time.time()

        # 控制更新周期为 10Hz
        if current_time - self.previous_time >= 0.1:
            # 获取车辆当前状态
            pos = self.vehicle.state["pos"]  # 车辆位置
            dir = self.vehicle.state["dir"]
            speed = self.vehicle.sensors["electrics"]["wheelspeed"]


            # 打印车辆信息
            print("The vehicle position is:", pos)
            print("The vehicle dir is:", dir)
            print("The vehicle speed is:", speed)

            # 提取最近的轨迹点和下一个轨迹点
            # target_pos=[0,-30]
            current_way_point = self.get_closest_traj_point(pos)
            next_way_point = self.get_next_traj_point(current_way_point)

            lookahead_distance = 20
            lookahead_point= []
            if(self.distance(pos, next_way_point) > 0.1):

                lookahead_point = self.calculate_lookahead_point(pos,lookahead_distance, next_way_point)

                steering_angle = self.calculate_steering_angle(pos, dir, lookahead_point)

                # 根据角度调整车辆转向，假设最大转向角为1.0
                steering = max(-1.0, min(1.0, steering_angle / math.pi))
                steering = -steering
                # 控制油门，假设油门强度为 0.8（可以根据需求调整）
                throttle = 0.2
                # 控制刹车，当目标航点距离较近时减速
                brake = 0.0
                if self.distance(pos, next_way_point) < 1.0:
                    brake = 0.5  # 如果接近目标点，开始刹车
                # 假设车辆在1档行驶
                gear = 2

                # 调用控制方法，传递计算得到的控制参数
                self.vehicle.control(
                    steering=steering,
                    throttle=throttle,
                    brake=brake,
                    parkingbrake=None,  # 假设不需要使用手刹
                    clutch=None,  # 假设不需要调整离合器
                    gear=gear
                )

                print(f"Steering: {steering}, Throttle: {throttle}, Brake: {brake}, Gear: {gear}")

            # 打印最近轨迹点和下一个轨迹点
            print("Closest Trajectory Point:", current_way_point)
            print("Next Trajectory Point:", next_way_point)
            # 更新可视化
            self.update_visualization(pos,current_way_point, next_way_point,lookahead_point)
            # 更新上一帧的时间
            self.previous_time = current_time

    def update_visualization(self, pos,current_way_point, next_way_point,lookahead_point):
        """
        更新轨迹、当前位置和下一个轨迹点的可视化
        """
        # 确保从轨迹数据中提取位置坐标
        traj_x = [entry["data"]["position"][0] for entry in self.traj]  # 提取所有轨迹点的 x 坐标
        traj_y = [entry["data"]["position"][1] for entry in self.traj]  # 提取所有轨迹点的 y 坐标

        # 更新轨迹线
        self.line_traj.set_data(traj_x, traj_y)

        if pos:
            self.scatter_current_pose.set_data([pos[0]], [pos[1]])

        if lookahead_point:
            self.scatter_look_ahead.set_data([lookahead_point[0]],[lookahead_point[1]])

        # 更新车辆当前位置
        if current_way_point:
            # 确保 current_point 是可迭代对象 (x, y)
            self.scatter_current_way_point.set_data([current_way_point[0]], [current_way_point[1]])

        # 更新下一个轨迹点
        if next_way_point:
            # 确保 next_point 是可迭代对象 (x, y)
            self.scatter_next_way_point.set_data([next_way_point[0]], [next_way_point[1]])

        # 刷新图形
        self.fig.canvas.draw_idle()  # 使用 draw_idle 来刷新图形
        self.fig.savefig("vis.png")

    def start_visualization(self):
        """
        启动可视化更新
        """
        plt.ion()  # 开启交互模式
        while True:
            self.update()
            time.sleep(0.1)  # 控制更新频率






if __name__ == "__main__":
    random.seed(1703)
    set_up_simple_logging()

    beamng = BeamNGpy("localhost", 25252, home='C:\Program Files (x86)\BeamNG.tech.v0.34.2.0')
    bng = beamng.open(launch=True)

    scenario = Scenario("smallgrid", "beamngpy_feature_overview")
    vehicle = Vehicle("ego", model="etk800", color="White", license="ZJU")
    electrics = Electrics()
    vehicle.sensors.attach("electrics", electrics)

    scenario.add_vehicle(vehicle)
    scenario.make(beamng)

    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()

    traj = []
    with open("vehicle_data.json", "r") as f:
        traj = json.load(f)

    # 设置MPC控制器
    mpc_controller = MPCController()

    # 创建车辆控制器
    vehicle_controller = VehicleController(beamng, vehicle, traj, mpc_controller)


    # 模拟过程
    while True:
        vehicle_controller.update()
        time.sleep(0.1)
