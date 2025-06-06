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
config_path = os.path.dirname(__file__)+"/../../../Environment/environment/env5/Navigation.sumocfg"  # Navigation task

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class Traffic_Env(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(self, attack=False, adv_steps=2, eval=False, use_gui=False, render_mode=None,random_seed = False):
        self.AutoCarID = 'Auto'
        self.maxDistance = 200.0
        self.maxSpeed = 30.0
        self.max_angle = 360.0
        self.x_goal = 2100.0
        self.y_goal = 150.0
        self.max_dis_navigation = 2500.0
        self.max_acc = 7.6
        self.reset_times = 0

        # only work for attack
        self.adv_steps = adv_steps
        self.attack_remain = adv_steps
        self.attack = attack
        self.eval = eval
        # For traci multi-client support
        self.label = str(Traffic_Env.CONNECTION_LABEL)
        Traffic_Env.CONNECTION_LABEL += 1
        self.random = random_seed
        if self.random:
            self.sumo_seed = 'random'
        elif self.eval:
            self.sumo_seed = 1
        else:
            self.sumo_seed = 0
        # define dims of action space
        if self.attack:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            # define dims of state space
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            # define dims of state space
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        # GUI
        self.use_gui = use_gui
        self.render_mode = render_mode
        if self.use_gui or self.render_mode is not None:
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')
        self.virtual_display = (1500, 1000)
    def raw_obs(self, vehicle_params):  # dimension: 24+5
        obs = []

        if self.AutoCarID in vehicle_params:
            zone = [[[],[],[]] for _ in range(6)]

            ego_veh_x, ego_veh_y = traci.vehicle.getPosition(self.AutoCarID)
            dis_goal_ego = np.linalg.norm(np.array([self.x_goal-ego_veh_x, self.y_goal-ego_veh_y]))

            obs_light = traci.vehicle.getNextTLS(self.AutoCarID)
            if obs_light is ():
                dis2light = self.maxDistance
                red_light = 0.0
            else:
                dis2light = min(obs_light[0][2], self.maxDistance)
                if obs_light[0][3] is 'r':
                    red_light = 1.0
                else:
                    red_light = 0.0

            for VehID in vehicle_params:
                veh_x, veh_y = traci.vehicle.getPosition(VehID)  # position, X & Y
                dis = np.linalg.norm(np.array([veh_x-ego_veh_x, veh_y-ego_veh_y]))

                if VehID != self.AutoCarID and dis < self.maxDistance:
                    angle = math.degrees(math.atan2(veh_y-ego_veh_y, veh_x-ego_veh_x))

                    if 0 <= angle < math.degrees(math.atan2(3**0.5, 1)): # 0~60
                        zone[0][0].append(VehID)
                        zone[0][1].append(dis)
                        zone[0][2].append(angle)
                    elif math.degrees(math.atan2(3**0.5, 1)) <= angle < math.degrees(math.atan2(3**0.5, -1)): # 60~120
                        zone[1][0].append(VehID)
                        zone[1][1].append(dis)
                        zone[1][2].append(angle)
                    elif math.degrees(math.atan2(3**0.5, -1)) <= angle < 180: # 120~180
                        zone[2][0].append(VehID)
                        zone[2][1].append(dis)
                        zone[2][2].append(angle)
                    elif -180 <= angle < math.degrees(math.atan2(-3**0.5, -1)): # -180~-120
                        zone[3][0].append(VehID)
                        zone[3][1].append(dis)
                        zone[3][2].append(angle)
                    elif math.degrees(math.atan2(-3**0.5, -1)) <= angle < math.degrees(math.atan2(-3**0.5, 1)): # -120~-60
                        zone[4][0].append(VehID)
                        zone[4][1].append(dis)
                        zone[4][2].append(angle)
                    else: # -60~0
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
            obs.append(dis2light)
            obs.append(red_light)
            obs.append(dis_goal_ego)
            info = {'x_position': ego_veh_x, 'y_position': ego_veh_y, 'infraction': 0.0,
                    'infraction_check': False, 'navigation_check': False,'reward': 0.0,'cost': 0.0, 'flag': False,'step': 0}

        else:
            obs = [self.maxDistance, 0.0, 0.0, 0.0, self.maxDistance, 0.0, 0.0, 0.0,self.maxDistance, 0.0, 0.0, 0.0,\
                   self.maxDistance, 0.0, 0.0, 0.0,self.maxDistance, 0.0, 0.0, 0.0, self.maxDistance, 0.0, 0.0, 0.0,\
                   0.0, 0.0, self.maxDistance, 0.0, self.max_dis_navigation]
            info = {'x_position': 0.0, 'y_position': 0.0, 'infraction': 0.0,
                    'infraction_check': False, 'navigation_check': False,'reward': 0.0,'cost': 0.0, 'flag': False,'step': 0}

        return obs, info

    def obs_to_state(self, vehicle_params):
        obs, info = self.raw_obs(vehicle_params)
        # print("raw_obs===>", obs)
        state = [obs[0]/self.maxDistance, obs[1]/self.max_angle, obs[2]/self.maxSpeed, obs[3]/self.max_angle,\
                 obs[4]/self.maxDistance, obs[5]/self.max_angle, obs[6]/self.maxSpeed, obs[7]/self.max_angle,\
                 obs[8]/self.maxDistance, obs[9]/self.max_angle, obs[10]/self.maxSpeed, obs[11]/self.max_angle,\
                 obs[12]/self.maxDistance, obs[13]/self.max_angle, obs[14]/self.maxSpeed, obs[15]/self.max_angle,\
                 obs[16]/self.maxDistance, obs[17]/self.max_angle, obs[18]/self.maxSpeed, obs[19]/self.max_angle,\
                 obs[20]/self.maxDistance, obs[21]/self.max_angle, obs[22]/self.maxSpeed, obs[23]/self.max_angle,\
                 obs[24]/self.maxSpeed, obs[25]/self.max_angle, obs[26]/self.maxDistance, obs[27], obs[28]/self.max_dis_navigation]

        if self.attack:
            state.append(self.attack_remain / self.adv_steps)
            state.append(0)
        return state, info

    def get_reward(self, vehicle_params):
        cost = 0.0
        infraction = 0.0
        infraction_check = False
        navigation_check = False
        done = False

        raw_obs, _ = self.raw_obs(vehicle_params)
        dis_fr = raw_obs[0]
        dis_f = raw_obs[4]
        dis_fl = raw_obs[8]
        dis_rl = raw_obs[12]
        dis_r = raw_obs[16]
        dis_rr = raw_obs[20]
        dis_sides = [dis_fr, dis_fl, dis_rl, dis_rr]
        v_ego = raw_obs[24]
        dis2light = raw_obs[26]
        red_light = raw_obs[27]
        dis_goal_ego = raw_obs[28]

        # efficiency
        reward = v_ego/self.maxSpeed

        # safety
        collision_value = self.check_collision(dis_f, dis_r, dis_sides, vehicle_params)
        if collision_value is True:
            cost = 1.0
            done = True

        # infraction
        if (red_light == 1.0) and (dis2light < 15):
            infraction = 1.0
            infraction_check = True
            done = True
            print("+++>infraction:", infraction_check, red_light, dis2light)

        # navigation
        if self.AutoCarID in vehicle_params and (
                dis_goal_ego < 15.0 or traci.vehicle.getLaneID(self.AutoCarID) == "E19_0"):
                navigation = 100.0
                navigation_check = True
                done = True
                print(">>>>>>Touch down!!!")
        else:
            navigation = -np.log(1.0 + dis_goal_ego / self.max_dis_navigation) - 1.0

        return reward-cost-infraction+navigation, collision_value, cost, infraction_check, infraction, navigation_check, done, reward

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
            action, mask = action_a[:2].item(), action_a[2].item()
            if mask:
                self.attack_remain -= 1
        else:
            action = action_a
        #print(action_a,action)
        acc, control_steer = action[0].item(), action[1].item()
        control_acc = self.max_acc * acc

        current_lane_id = traci.vehicle.getLaneID(self.AutoCarID)
        edge_id = traci.lane.getEdgeID(current_lane_id)
        max_lane_index = traci.edge.getLaneNumber(edge_id) - 1
        lane_index = traci.vehicle.getLaneIndex(self.AutoCarID)

        if -0.5 <= control_steer < 0.0:
            lane_change = 1  # left changing lane
        elif 0.0 <= control_steer < 0.5:
            lane_change = -1  # right changing lane
        else:
            lane_change = 0  # go straight

        if lane_index == max_lane_index and lane_change == 1:
            lane_change = 0

        if lane_index == 0 and lane_change == -1:
            lane_change = 0

        if edge_id == 'E18':
            traci.vehicle.changeLane(self.AutoCarID, lane_index + lane_change, 0)
            # print(">>>>>>", current_lane_id, edge_id)

        traci.vehicle.setSpeed(self.AutoCarID, max(traci.vehicle.getSpeed(self.AutoCarID) + control_acc, 0.001))
        traci.simulationStep()

        # Get the new vehicle parameters
        new_vehicle_params = traci.vehicle.getIDList()
        reward_cost, collision_value, cost, infraction_check, infraction, navigation_check, done,reward = self.get_reward(new_vehicle_params)
        next_state, info = self.obs_to_state(new_vehicle_params)

        info['reward'] = reward
        info['infraction'] = infraction
        info['cost'] = cost
        info['infraction_check'] = infraction_check
        info['navigation_check'] = navigation_check

        if self.attack and mask and collision_value:
            info['flag'] = True
        if self.attack and not self.eval:
            if self.attack_remain == 0:
                print("===>Checker-3: Attack times run out!")
                return np.array(next_state, dtype=np.float32), cost, collision_value, True, info
            else:
                return np.array(next_state, dtype=np.float32), cost, collision_value, False, info
        else:
            return np.array(next_state, dtype=np.float32), reward_cost, collision_value, False, info

    def reset(self, seed=None, options=None):
        # dom = xml.dom.minidom.parse(config_path)
        # root = dom.documentElement
        # random_seed_element = root.getElementsByTagName("seed")[0]
        self.attack_remain = self.adv_steps
        if self.random:
            self.sumo_seed = 'random'
        else:
            if self.reset_times % 2 == 0:
                self.sumo_seed = "%d" % self.reset_times
                if self.eval:
                    self.sumo_seed = "%d" % (self.reset_times + 1)
        # if self.reset_times % 2 == 0:
        #     random_seed = "%d" % self.reset_times
            #random_seed_element.setAttribute("value", random_seed)

        # with open(config_path, "w") as file:
        #     dom.writexml(file)
        self.start()
        #traci.load(["-c", config_path])
        print('Resetting the layout!!!!!!', self.reset_times)
        self.reset_times += 1

        AutoCarAvailable = False
        while AutoCarAvailable == False:
            traci.simulationStep()
            VehicleIds = traci.vehicle.getIDList()
            if self.AutoCarID in VehicleIds:
                AutoCarAvailable = True

        # Just check if the auto car still exisits and that there has not been any collision
        for VehId in VehicleIds:
            if VehId == self.AutoCarID:
                traci.vehicle.setSpeedMode(VehId, 22)
                traci.vehicle.setLaneChangeMode(VehId, 1)  # Disable automatic lane changing

        initial_state, info = self.obs_to_state(VehicleIds)

        return np.array(initial_state, dtype=np.float32), info

    def close(self):
        traci.close()

    def start(self, gui=False):
        self.sumoBinary = checkBinary('sumo-gui') if self.use_gui else checkBinary('sumo')
        sumo_cmd = [self.sumoBinary, "-c", config_path, "--collision.check-junctions", "true"]
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
        else:
            self.label = str(time.time())
            traci.start(sumo_cmd, label="init_connection" + self.label)
            traci.getConnection("init_connection" + self.label)

        if self.use_gui or self.render_mode is not None:
            traci.gui.DEFAULT_VIEW = "View #0"
            traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def render(self):
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame
        elif self.render_mode == "rgb_array":
            # get position of AutoCar
            x, y = traci.vehicle.getPosition(self.AutoCarID)
            # 设置偏移量和缩放级别来锁定视图
            traci.gui.setZoom("View #0", 1500)  # 设置缩放级别
            traci.gui.setOffset("View #0", x, y)  # 设置视图的偏移量（锁定到车辆位置）
            traci.gui.setSchema("View #0", "real world")
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            return np.array(img)


    def get_obs(self):
        return self.obs

    def set_obs(self, obs):
        self.obs = obs



