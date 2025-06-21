import numpy as np
import math
import traci
import gymnasium as gym
import numpy as np
import os
import sys
import math
import xml.dom.minidom
from gymnasium import spaces
from scipy.optimize import fsolve
from sympy.integrals.intpoly import cross_product


def calculate_velocity_components(speed, heading_angle):
    """
    根据航向角和合速度计算横纵速度分量
    :param speed: 合速度 (m/s)
    :param heading_angle: 航向角 θ (弧度制)
    :return: 纵向速度 v_x, 横向速度 v_y
    """
    vx = speed * math.cos(heading_angle)
    vy = speed * math.sin(heading_angle)
    return vy, vx


def compute_deltarate(vehicle1_id, vehicle2_id, delta, time_step, angle, last_angle):
    """
        计算接近率
    """
    # 获取车辆状态数据
    pos1 = np.array(traci.vehicle.getPosition(vehicle1_id))
    pos2 = np.array(traci.vehicle.getPosition(vehicle2_id))
    angle1 = np.radians(traci.vehicle.getAngle(vehicle1_id))
    angle2 = np.radians(traci.vehicle.getAngle(vehicle2_id))
    speed1 = np.array(calculate_velocity_components(traci.vehicle.getSpeed(vehicle1_id), angle1))
    speed2 = np.array(calculate_velocity_components(traci.vehicle.getSpeed(vehicle2_id), angle2))
    acc1 = np.array(calculate_velocity_components(traci.vehicle.getAcceleration(vehicle1_id), angle1))
    acc2 = np.array(calculate_velocity_components(traci.vehicle.getAcceleration(vehicle2_id), angle2))

    # 计算位置差向量
    r_1_2 = pos2 - pos1

    # 单位化方向向量
    direction = r_1_2 / np.linalg.norm(r_1_2)

    # 最短距离方向上的投影速度和加速度
    v_1_2 = np.dot(speed1, direction)
    a_1_2 = np.dot(acc1, direction)
    v_2_1 = np.dot(speed2, -direction)
    a_2_1 = np.dot(acc2, -direction)
    costheta1 = np.dot(speed1, direction) / traci.vehicle.getSpeed(vehicle1_id)

    # 横向加速度/纵向速度
    # 计算偏转角速度
    last_angle = np.radians(last_angle)
    dot_theta1 = angle1 - last_angle
    dot_theta2 = 0
    # dot_theta1 = np.sin(angle1) / time_step
    # dot_theta2 = np.sin(angle2) / time_step

    # Rel 操作
    rel_v = v_1_2 + v_2_1
    rel_a = a_1_2 + a_2_1
    rel_theta = np.radians(dot_theta1 + dot_theta2)
    # print('rel_theta', rel_theta)

    # 计算 接近率
    delta_rate = rel_v + rel_a * time_step + rel_theta * delta

    # 用公式计算 错误 暂时废除
    angle = np.radians(angle)
    delta_rate1 = math.cos(angle) * (traci.vehicle.getSpeed(vehicle1_id) -
                                     traci.vehicle.getSpeed(vehicle2_id) +
                                     (traci.vehicle.getAcceleration(vehicle1_id) - traci.vehicle.getAcceleration(
                                         vehicle2_id)) * time_step) + (rel_theta * delta)
    # print(angle,delta_rate1,rel_v,rel_a,angle1,last_angle)

    return delta_rate, rel_v, rel_a, rel_theta, direction, costheta1


def calculate_ACT(vehicle1_id, AutoCarID, speed, delta, angle, last_angle):
    """
        计算ACT ACT = distance / deltarate 距离/接近率 及 可碰撞加速度区间

        vehicle1_id:目标车辆
        speed: 上个时刻自动驾驶车辆的速度
        delta:最短距离
        angle:两车夹角
        last_angle:上个时刻自动驾驶车辆的航向角

        用法：
        act = ACT.calculate_ACT(self,VehID,dis,angle,self.angles[-1] if self.angles else 0)
    """
    # 计算接近率
    deltarate, relv, rela, relt, d, costheta1 = compute_deltarate(AutoCarID, vehicle1_id, delta, 1, angle,
                                                                  last_angle)

    # print("checking vehicleID:{},dis:{},speed:{},deltarate:{},ACT:{},pos:{}".format(vehicle1_id, delta, traci.vehicle.getSpeed(vehicle1_id),
    #                                                                          deltarate,delta/deltarate,
    #                                                                traci.vehicle.getPosition(vehicle1_id)))
    if deltarate > 0 and checkOnCollision(vehicle1_id):
        # 接近率大于0返回ACT
        # 降低ACT的加速度阈值
        amin, amax = AccRangeForCollision(costheta1, delta, AutoCarID, vehicle1_id)
        real_amin = max(-7.6, 0 - speed)
        real_amax = min(7.6, 15 - speed)
        intersection = get_intersection(amin, amax, real_amin, real_amax)
        if delta / deltarate < 5 and intersection:
            return delta / deltarate, intersection
        else:
            return 1e6, None
    else:
        # 否则返回无穷
        return 1e6, None


def calculate_ACT_v2(vehicle1_id, AutoCarID, speed, delta, angle, last_angle):
    """
        计算ACT ACT = distance / deltarate 距离/接近率

        vehicle1_id:目标车辆
        speed: 上个时刻自动驾驶车辆的速度
        delta:最短距离
        angle:两车夹角
        last_angle:上个时刻自动驾驶车辆的航向角

        用法：
        act = ACT.calculate_ACT(self,VehID,dis,angle,self.angles[-1] if self.angles else 0)
    """
    # 计算接近率
    deltarate, relv, rela, relt, d, costheta1 = compute_deltarate(vehicle1_id, AutoCarID, delta, 1, angle,
                                                                  last_angle)

    # print("checking vehicleID:{},dis:{},speed:{},deltarate:{},ACT:{},pos:{}".format(vehicle1_id, delta, traci.vehicle.getSpeed(vehicle1_id),
    #                                                                          deltarate,delta/deltarate,
    #                                                                traci.vehicle.getPosition(vehicle1_id)))
    # if deltarate > 0 and checkOnCollision(vehicle1_id):
    #     # 接近率大于0返回ACT
    #     # 降低ACT的加速度阈值
    #     amin, amax = AccRangeForCollision(costheta1, delta, AutoCarID, vehicle1_id)
    #     real_amin = max(-7.6, 0 - speed)
    #     real_amax = min(7.6, 15 - speed)
    #     intersection = get_intersection(amin, amax, real_amin, real_amax)
    #     if delta / deltarate < 5 and intersection:
    #         return delta / deltarate, intersection
    #     else:
    #         return 1e6, None
    # else:
    #     # 否则返回无穷
    #     return 1e6, None

    # if delta == 0:
    #     print('delta is 0')
    # elif deltarate == 0:
    #     print('deltarate is 0', delta)

    if deltarate > 0 and checkOnCollisionv2(vehicle1_id):
    # if deltarate > 0:
        if delta / deltarate < 3:
            return delta / deltarate
        else:
            return 10
    else:
        return 25




def get_intersection(pred_min, pred_max, real_min, real_max):
    start = max(pred_min, real_min)
    end = min(pred_max, real_max)
    if start <= end:
        return [start, end]
    else:
        return None



def ifOnCollisin(vehicle1_id, vehicle2_id):
    """
            论文中判断两车是否处于碰撞过程的方法 鸡肋 暂时未用
    """
    # print("checking on collision")
    x1, y1 = traci.vehicle.getPosition(vehicle1_id)
    x2, y2 = traci.vehicle.getPosition(vehicle2_id)
    angle1 = np.radians(traci.vehicle.getAngle(vehicle1_id))
    angle2 = np.radians(traci.vehicle.getAngle(vehicle2_id))
    L1 = traci.vehicle.getLength(vehicle1_id)
    L2 = traci.vehicle.getLength(vehicle2_id)
    W1 = traci.vehicle.getWidth(vehicle1_id)
    W2 = traci.vehicle.getWidth(vehicle2_id)

    # 计算最近点对
    Corners1 = calculate_corners_with_labels(x1, y1, angle1, L1, W1)
    Corners2 = calculate_corners_with_labels(x2, y2, angle2, L2, W2)
    closest_pair, closest_dis, corner_type = closest_corners(Corners1, Corners2)

    # print(vehicle1_id,L1,W1,math.sin(angle1),math.cos(angle1),x1,y1,Corners1,vehicle2_id,angle2,x2,y2,Corners2)

    # 判断两车相向或同向
    if corner_type[0] < 2:
        type1 = -1  # 车头
    else:
        type1 = 1  # 车尾
    if corner_type[1] < 2:
        type2 = -1
    else:
        type2 = 1
    # print("type1,type2",type1,type2)
    if type1 * type2 == -1:  # 同向
        # 判断前后车
        if type1 == -1:
            rearID = vehicle1_id
            frontID = vehicle2_id
        else:
            rearID = vehicle2_id
            frontID = vehicle1_id
        # 判断RedLine是否过另一最近点的两条临边
        xr, yr = traci.vehicle.getPosition(rearID)
        angler = traci.vehicle.getAngle(rearID)
        RedLine_m, RedLine_c = parallel_line_through_corner(xr, yr, angler)
        # print(angler,RedLine_m,RedLine_c)
        if frontID == vehicle1_id:
            xf, yf = closest_pair[0]
            xf1, yf1 = Corners1[(corner_type[0] - 1) % 4]
            xf2, yf2 = Corners1[(corner_type[0] + 1) % 4]
        else:
            xf, yf = closest_pair[1]
            xf1, yf1 = Corners2[(corner_type[1] - 1) % 4]
            xf2, yf2 = Corners2[(corner_type[1] + 1) % 4]

        if np.sign(yf - xf * RedLine_m + RedLine_c) * np.sign(yf1 - xf1 * RedLine_m + RedLine_c) <= 0:
            return True
        elif np.sign(yf - xf * RedLine_m + RedLine_c) * np.sign(yf2 - xf2 * RedLine_m + RedLine_c) <= 0:
            return True
        else:
            return False
    else:
        if check_intersection(closest_pair[0], closest_pair[1], vehicle1_id, vehicle2_id)[0]:
            return True
        else:
            return False


def AccRangeForCollision(costheta, delta, vehicle1_id, vehicle2_id):
    """
    计算ACT在某一范围时自动驾驶车辆加速度的取值
    """
    # Revised：1/ACT = m * a1 - c
    # m = costheta/delta
    # c = m * a2 + (v1-v2)* m + (theta_dot1+theta_dot2)
    theta_dot1 = 0
    theta_dot2 = 0
    v1 = traci.vehicle.getSpeed(vehicle1_id)
    v2 = traci.vehicle.getSpeed(vehicle2_id)
    a2 = traci.vehicle.getAcceleration(vehicle2_id)
    # theta = np.radians(theta)

    m = costheta / delta
    C = m * a2 + (v1 - v2) * m + theta_dot1 + theta_dot2

    xstar = (1 + C) / m
    xstar_ = (1 / 1 + C) / m
    if m > 0:
        # a_min = xstar
        # a_max = 7.6
        a_min = xstar_
        a_max = xstar
    else:
        # a_min = -7.6
        # a_max = xstar
        a_min = xstar
        a_max = xstar_
    # # Calculate x for y=0 and y=1
    # x1 = -C / m  # x when y = 0
    # x2 = (1 - C) / m  # x when y = 1
    #
    # # Determine the range
    # a_min = min(x1, x2)
    # a_max = max(x1, x2)

    return a_min, a_max



def calculate_corners_with_labels(x0, y0, theta, L, W):
    """
    计算车辆拐角坐标并区分前后位置。

    参数:
    x0, y0 : float
        车辆前保险杠中央的坐标 (x, y)
    theta : float
        偏航角 (单位: 弧度)
    L : float
        车辆长度 (前后长度)
    W : float
        车辆宽度 (左右宽度)

    返回:
    corners_with_labels : dict
        包含拐角坐标及其标记的字典，例如：
        {
            "front_left": (x1, y1),
            "front_right": (x2, y2),
            "rear_left": (x3, y3),
            "rear_right": (x4, y4)
        }
    """
    theta = np.radians(theta)
    x0 = x0 - (L / 2) * math.cos(theta)
    y0 = y0 - (L / 2) * math.sin(theta)
    # 前左拐角
    x1 = x0 + (L / 2) * math.cos(theta) + (W / 2) * math.sin(theta)
    y1 = y0 + (L / 2) * math.sin(theta) - (W / 2) * math.cos(theta)
    # 前右拐角
    x2 = x0 + (L / 2) * math.cos(theta) - (W / 2) * math.sin(theta)
    y2 = y0 + (L / 2) * math.sin(theta) + (W / 2) * math.cos(theta)
    # 后左拐角
    x3 = x0 - (L / 2) * math.cos(theta) + (W / 2) * math.sin(theta)
    y3 = y0 - (L / 2) * math.sin(theta) - (W / 2) * math.cos(theta)
    # 后右拐角
    x4 = x0 - (L / 2) * math.cos(theta) - (W / 2) * math.sin(theta)
    y4 = y0 - (L / 2) * math.sin(theta) + (W / 2) * math.cos(theta)

    corners_with_labels = [
        (x1, y1),
        (x2, y2),
        (x3, y3),
        (x4, y4)
    ]

    return corners_with_labels


def parallel_line_through_corner(x_c, y_c, theta):
    """
    计算平行于车辆长边且过给定点 (x_c, y_c) 的直线方程。

    参数:
    x_c, y_c : float
        给定点 (C1) 的坐标
    theta : float
        车辆偏航角 (单位: 弧度)

    返回:
    mc : 直线方程的参数
    """
    # y=mx+c
    theta = math.radians(theta)
    m = math.tan(theta)
    # print('m',m,theta,math.sin(theta),math.cos(theta))
    c = y_c - m * x_c
    return m, c


def distance(x1, y1, x2, y2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def closest_corners(V1_corners, V2_corners):
    """
    计算两辆车拐角中距离最近的两个点

    参数:
    V1_corners : list of tuples
        车辆V1的四个拐角坐标 [(x11, y11), (x12, y12), (x13, y13), (x14, y14)]
    V2_corners : list of tuples
        车辆V2的四个拐角坐标 [(x21, y21), (x22, y22), (x23, y23), (x24, y24)]

    返回:
    tuple : ((x1, y1), (x2, y2), min_distance)
        最短距离的两个点坐标及该距离
    """
    min_distance = float('inf')
    closest_points = None
    ind1 = -1
    ind2 = -1
    # 遍历V1的每个拐角和V2的每个拐角，计算最短距离
    for (x1, y1) in V1_corners:
        ind1 += 1
        inds2 = -1
        for (x2, y2) in V2_corners:
            inds2 += 1
            dist = distance(x1, y1, x2, y2)
            if dist < min_distance:
                min_distance = dist
                closest_points = ((x1, y1), (x2, y2))
                ind2 = inds2

    return closest_points, min_distance, (ind1, ind2)


def checkOnCollision(vehicleID):
    """
            仅适用于Env3，通过车辆所处的位置朴素的判断该车辆与自动驾驶车辆是否有碰撞可能
    """
    vp = traci.vehicle.getPosition(vehicleID)
    ap = traci.vehicle.getPosition('Auto')
    if vp[1] == 4.8:  # 车1
        if vp[0] > 1.6 and ap[1] < 4.8:
            return True
        if ap[1] == 4.8:
            return True
        else:
            return False
    elif vp[1] == -4.8:  # 车3
        if ap[1] < -4.8 and vp[0] < 1.6:
            return True
        else:
            return False
    elif vp[0] == -4.8:  # 车2
        if 1.6 > ap[0] > -4.8:
            return True
        else:
            return False


def checkOnCollisionv2(vehicleID):
    """
        通过航向角判断是否处于碰撞过程
    """
    p2 = traci.vehicle.getPosition(vehicleID)
    p1 = traci.vehicle.getPosition('Auto')
    angle2 = math.radians(traci.vehicle.getAngle(vehicleID))
    angle1 = math.radians(traci.vehicle.getAngle('Auto'))
    d1 = (math.sin(angle1),math.cos(angle1))
    d2 = (math.sin(angle2),math.cos(angle2))
    #print(p1,p2,d1,d2)
    flag = check_lines_intersect_or_coincide(p1,d1,p2,d2)

    # #右偏15°
    # d2_right = rotate_vector(d2,-np.radians(15))
    # flag_right = check_lines_intersect_or_coincide(p1,d2_right,p2,d1)
    #
    # #左偏15°
    # d2_left = rotate_vector(d2,np.radians(15))
    # flag_left = check_lines_intersect_or_coincide(p1,d2_left,p2,d1)
    #
    # #print(vehicleID,flag,d1,d2)
    # if flag or flag_right or flag_left:
    #     return True
    # else:
    #     return False
    return flag

def check_lines_intersect_or_coincide(A, a, B, b):
    """
    判断从点A沿方向向量a出发的直线与从点B沿方向向量b出发的直线是否相交或重合.

    :param A: 点A的坐标 (x1, y1)
    :param a: 方向向量a (a1, a2)
    :param B: 点B的坐标 (x2, y2)
    :param b: 方向向量b (b1, b2)
    :return: True如果有碰撞可能 否则False
    """
    # 计算方向向量的叉积
    cross_product_norm = np.linalg.norm(np.cross(a, b))
    # print(cross_product_norm)
    # cross_product = a[0] * b[1] - a[1] * b[0]
    #print('cross_pro',cross_product)
    if 0 <= cross_product_norm <= 1e-10:
        # 两直线平行或重合
        # 检查点A是否在直线B上
        vector_AB = np.array(B) - np.array(A)
        dot_product = np.dot(vector_AB, b)
        magnitude_AB = np.linalg.norm(vector_AB)
        magnitude_b = np.linalg.norm(b)
        #print('ss',dot_product-magnitude_AB*magnitude_b)
        if 0 <= dot_product-(magnitude_AB * magnitude_b)<1:
            return True
        else:
            return False
    else:
        # # 两直线不平行，计算交点
        # t = ((B[0] - A[0]) * b[1] - (B[1] - A[1]) * b[0]) / cross_product
        # s = ((B[0] - A[0]) * a[1] - (B[1] - A[1]) * a[0]) / cross_product
        # if t >= 0 and s >= 0:
        #     return True
        # else:
        #     return False
        return True


def rotate_vector(v, angle):
    """
    旋转向量v角度angle（以弧度为单位）.

    :param v: 向量 (x, y)
    :param angle: 旋转角度（弧度）
    :return: 旋转后的向量
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, v)