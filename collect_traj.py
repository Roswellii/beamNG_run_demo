import numpy as np
import time
import json
from beamngpy import BeamNGpy, Scenario, Vehicle
from scipy.optimize import minimize
import random
from beamngpy.sensors import Electrics

# 设置随机种子
random.seed(1703)

# 初始化日志配置
from beamngpy import set_up_simple_logging

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
    def __init__(self, beamng_connection, vehicle):
        self.beamng_connection = beamng_connection

        self.previous_time = time.time()
        self.vehicle= vehicle


    def update(self):
        self.vehicle.sensors.poll()
        current_time = time.time()
        if current_time - self.previous_time >= 0.1:  # 控制周期为10Hz
            # 获取车辆当前状态
            pos = self.vehicle.state["pos"]  # 车辆位置
            dir =  self.vehicle.state["dir"]
            speed = self.vehicle.sensors["electrics"]["wheelspeed"]

            # 打印车辆信息
            print("The vehicle position is:", pos)
            print("The vehicle dir is:", dir)
            print("The vehicle speed is:", speed)

            # 准备要写入JSON的数据
            data = {
                "timestamp": current_time,
                "data": {
                    "position": pos,
                    "direction": dir,
                    "speed": speed
                }
            }

            # 打开JSON文件并将数据追加到文件中
            try:
                # 如果文件不存在，则创建文件并初始化为一个空列表
                try:
                    with open("vehicle_data.json", "r") as f:
                        all_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_data = []

                # 将新数据追加到列表
                all_data.append(data)

                # 写入更新后的数据到文件
                with open("vehicle_data.json", "w") as f:
                    json.dump(all_data, f, indent=4)

            except IOError as e:
                print(f"Error writing to file: {e}")

            #
            # current_state = pos + [0]  # 当前状态（假设方向为0）
            #
            # 获取未来轨迹点
           #  trajectory = [self.trajectory_loader.get_next_point(position)] * self.mpc_controller.prediction_horizon
            #
            # # 获取MPC优化的控制指令
            # control_inputs = self.mpc_controller.get_control(current_state, trajectory)
            #
            # # 在这里，你可以调用BeamNG来设置车速和方向
            # for i in range(self.mpc_controller.prediction_horizon):
            #     speed, steering_angle = control_inputs[i * 2:i * 2 + 2]
            #     self.vehicle.control(throttle=speed, steering=steering_angle)
            #
            # self.previous_time = current_time


if __name__ == "__main__":
    random.seed(1703)
    set_up_simple_logging()

    beamng = BeamNGpy("localhost", 25252, home='C:\Program Files (x86)\BeamNG.tech.v0.34.2.0')
    bng = beamng.open(launch=True)

    scenario = Scenario(level="italy",  name="beamngpy_feature_overview")
    vehicle = Vehicle("ego", model="etk800", color="White", license="ZJU")
    electrics = Electrics()
    vehicle.sensors.attach("electrics", electrics)
    scenario.add_vehicle(vehicle,pos=(245.11, -906.94, 247.46), rot_quat=(0.0010, 0.1242, 0.9884, -0.0872))
    # scenario.add_vehicle(vehicle)
    scenario.make(beamng)

    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()




    # 创建车辆控制器
    vehicle_controller = VehicleController(beamng, vehicle)


    # 模拟过程
    while True:
        vehicle_controller.update()
        time.sleep(0.1)
