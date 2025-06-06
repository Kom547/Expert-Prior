from __future__ import absolute_import
from __future__ import print_function

import gymnasium as gym
import numpy as np
import os
import sys
import math
import xml.dom.minidom
from gymnasium import spaces

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

gui = True
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')
config_path = os.path.dirname(__file__) + "/../../../Environment/environment/env7/highway.sumocfg"  # 高速巡航任务

class Traffic_Env(gym.Env):

    def __init__(self, attack=False, adv_steps=2, eval=False):

        # 环境参数
        self.state_dim = 26
        self.action_dim = 2  # 加速度和转向
        self.maxDistance = 200.0
        self.maxSpeed = 30.0  # 调整为 30.0 以匹配提供的网络文件
        self.max_angle = 360.0
        self.max_acc = 7.6
        self.AutoCarID = 'Auto_vehicle'  # 根据 rou 文件调整为 'Auto_vehicle'
        self.reset_times = 0
        self.obs=None

        # 攻击相关参数
        self.adv_steps = adv_steps
        self.attack_remain = adv_steps
        self.attack = attack
        self.eval = eval

        # 定义动作空间和观察空间
        if self.attack:
            # 攻击模式下，动作空间包含加速度、转向和攻击掩码
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            # 状态空间增加两个维度：攻击剩余步数和攻击掩码
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        else:
            # 非攻击模式下，动作空间仅包含加速度和转向
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            # 定义维度为26的状态空间
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

    def raw_obs(self, vehicle_params):
        obs = []

        if self.AutoCarID in vehicle_params:
            zone = [[[], [], []] for _ in range(6)]
            ego_veh_x, ego_veh_y = traci.vehicle.getPosition(self.AutoCarID)

            for VehID in vehicle_params:
                veh_x, veh_y = traci.vehicle.getPosition(VehID)  # 位置，X和Y
                dis = np.linalg.norm(np.array([veh_x - ego_veh_x, veh_y - ego_veh_y]))

                if VehID != self.AutoCarID and dis < self.maxDistance:
                    angle = math.degrees(math.atan2(veh_y - ego_veh_y, veh_x - ego_veh_x))

                    if 0 <= angle < math.degrees(math.atan2(3**0.5, 1)):  # 0~60
                        zone[0][0].append(VehID)
                        zone[0][1].append(dis)
                        zone[0][2].append(angle)
                    elif math.degrees(math.atan2(3**0.5, 1)) <= angle < math.degrees(math.atan2(3**0.5, -1)):  # 60~120
                        zone[1][0].append(VehID)
                        zone[1][1].append(dis)
                        zone[1][2].append(angle)
                    elif math.degrees(math.atan2(3**0.5, -1)) <= angle < 180:  # 120~180
                        zone[2][0].append(VehID)
                        zone[2][1].append(dis)
                        zone[2][2].append(angle)
                    elif -180 <= angle < math.degrees(math.atan2(-3**0.5, -1)):  # -180~-120
                        zone[3][0].append(VehID)
                        zone[3][1].append(dis)
                        zone[3][2].append(angle)
                    elif math.degrees(math.atan2(-3**0.5, -1)) <= angle < math.degrees(math.atan2(-3**0.5, 1)):  # -120~-60
                        zone[4][0].append(VehID)
                        zone[4][1].append(dis)
                        zone[4][2].append(angle)
                    else:  # -60~0
                        zone[5][0].append(VehID)
                        zone[5][1].append(dis)
                        zone[5][2].append(angle)

            for z in zone:
                if len(z[0]) == 0:
                    obs.append(self.maxDistance)
                    obs.append(0.0)
                    obs.append(0.0)
                    obs.append(0.0)
                else:
                    mindis_index = z[1].index(min(z[1]))
                    obs.append(min(z[1]))
                    obs.append(z[2][mindis_index])
                    obs.append(traci.vehicle.getSpeed(z[0][mindis_index]))
                    obs.append(traci.vehicle.getAngle(z[0][mindis_index]))

            obs.append(traci.vehicle.getSpeed(self.AutoCarID))
            obs.append(traci.vehicle.getAngle(self.AutoCarID))
            info = {'x_position': ego_veh_x, 'y_position': ego_veh_y, 'reward': 0.0, 'cost': 0.0, 'flag': False}
        else:
            obs = [self.maxDistance, 0.0, 0.0, 0.0, self.maxDistance, 0.0, 0.0, 0.0,
                   self.maxDistance, 0.0, 0.0, 0.0,
                   self.maxDistance, 0.0, 0.0, 0.0,
                   self.maxDistance, 0.0, 0.0, 0.0,
                   self.maxDistance, 0.0, 0.0, 0.0,
                   0.0, 0.0]
            info = {'x_position': 0.0, 'y_position': 0.0, 'reward': 0.0, 'cost': 0.0, 'flag': False}

        return obs, info

    def obs_to_state(self, vehicle_params):
        obs, info = self.raw_obs(vehicle_params)
        # print("raw_obs===>", obs)
        state = [
            obs[0] / self.maxDistance, obs[1] / self.max_angle, obs[2] / self.maxSpeed, obs[3] / self.max_angle,
            obs[4] / self.maxDistance, obs[5] / self.max_angle, obs[6] / self.maxSpeed, obs[7] / self.max_angle,
            obs[8] / self.maxDistance, obs[9] / self.max_angle, obs[10] / self.maxSpeed, obs[11] / self.max_angle,
            obs[12] / self.maxDistance, obs[13] / self.max_angle, obs[14] / self.maxSpeed, obs[15] / self.max_angle,
            obs[16] / self.maxDistance, obs[17] / self.max_angle, obs[18] / self.maxSpeed, obs[19] / self.max_angle,
            obs[20] / self.maxDistance, obs[21] / self.max_angle, obs[22] / self.maxSpeed, obs[23] / self.max_angle,
            obs[24] / self.maxSpeed, obs[25] / self.max_angle
        ]

        if self.attack:
            state.append(self.attack_remain/self.adv_steps)
        return state, info

    def get_reward_a(self, vehicle_params):
        cost = 0.0
        raw_obs, _ = self.raw_obs(vehicle_params)
        dis_fr = raw_obs[0]
        dis_f = raw_obs[4]
        dis_fl = raw_obs[8]
        dis_rl = raw_obs[12]
        dis_r = raw_obs[16]
        dis_rr = raw_obs[20]
        v_ego = raw_obs[24]
        dis_sides = [dis_fr, dis_fl, dis_rl, dis_rr]

        # 效率
        reward = v_ego / self.maxSpeed

        # 安全
        collision_value = self.check_collision(dis_f, dis_r, dis_sides, vehicle_params)
        if collision_value is True:
            cost = 1.0

        return reward - cost, collision_value, reward, cost

    def check_collision(self, dis_f, dis_r, dis_sides, vehicle_params):
        collision_value = False

        if (dis_f < 2.0) or (dis_r < 1.5) or (min(dis_sides) < 1.0):
            collision_value = True
            print("--->Checker-1: Collision!")
        elif self.AutoCarID not in vehicle_params:
            collision_value = True
            print("===>Checker-2: Collision!")

        return collision_value

    def step(self, action_a):
        if self.attack:
            # 动作空间为 (acc, steer, mask)
            acc, steer, mask = action_a[0].item(), action_a[1].item(), action_a[2].item()
            if mask:
                self.attack_remain -= 1
        else:
            # 动作空间为 (acc, steer)
            acc, steer = action_a[0], action_a[1]
            mask = 0  # 无攻击

        # 应用加速度
        control_acc = self.max_acc * acc
        current_speed = traci.vehicle.getSpeed(self.AutoCarID)
        traci.vehicle.setSpeed(self.AutoCarID, max(current_speed + control_acc, 0.001))

        # 应用转向（变道）
        current_lane_id = traci.vehicle.getLaneID(self.AutoCarID)
        edge_id = traci.lane.getEdgeID(current_lane_id)
        max_lane_index = traci.edge.getLaneNumber(edge_id) - 1
        lane_index = traci.vehicle.getLaneIndex(self.AutoCarID)

        if -0.5 <= steer < 0.0:
            lane_change = 1  # 左变道
        elif 0.0 <= steer < 0.5:
            lane_change = -1  # 右变道
        else:
            lane_change = 0  # 直行

        if lane_index == max_lane_index and lane_change == 1:
            lane_change = 0

        if lane_index == 0 and lane_change == -1:
            lane_change = 0

        if edge_id == 'E18':
            traci.vehicle.changeLane(self.AutoCarID, lane_index + lane_change, 0)
            # print(">>>>>>", current_lane_id, edge_id)

        traci.simulationStep()

        # 获取新的车辆参数
        new_vehicle_params = traci.vehicle.getIDList()
        reward_cost, collision_value, reward, cost = self.get_reward_a(new_vehicle_params)
        next_state, info = self.obs_to_state(new_vehicle_params)

        info['reward'] = reward
        info['cost'] = cost
        if self.attack and not self.eval:
            return np.array(next_state, dtype=np.float32), cost, collision_value, False, info
        else:
            return np.array(next_state, dtype=np.float32), reward_cost, collision_value, False, info

    def reset(self, seed=None, options=None):
        self.attack_remain = self.adv_steps
        dom = xml.dom.minidom.parse(config_path)
        root = dom.documentElement
        random_seed_element = root.getElementsByTagName("seed")[0]
        if self.reset_times % 2 == 0:
            random_seed = "%d" % self.reset_times
            random_seed_element.setAttribute("value", random_seed)

        with open(config_path, "w") as file:
            dom.writexml(file)

        traci.load(["-c", config_path])
        print('Resetting the layout!!!!!!', self.reset_times)
        self.reset_times += 1

        AutoCarAvailable = False
        while not AutoCarAvailable:
            traci.simulationStep()
            VehicleIds = traci.vehicle.getIDList()
            if self.AutoCarID in VehicleIds:
                AutoCarAvailable = True

        # 检查自车是否存在并且未发生碰撞
        for VehId in VehicleIds:
            if VehId == self.AutoCarID:
                traci.vehicle.setSpeedMode(VehId, 22)
                traci.vehicle.setLaneChangeMode(VehId, 0)  # 禁用自动变道

        initial_state, info = self.obs_to_state(VehicleIds)
        self.set_obs(initial_state)
        return np.array(initial_state, dtype=np.float32), info

    def close(self):
        traci.close()

    def start(self, gui=False):
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        traci.start([sumoBinary, "-c", config_path, "--collision.check-junctions", "--log", "sumo_output.log",
                     "--message-log", "sumo_output.log", "--error-log", "error.log", "--no-step-log", "True",
                     "--duration-log.disable", "true", "--no-warnings"])

    def get_obs(self):
        return self.obs

    def set_obs(self, obs):
        self.obs = obs
