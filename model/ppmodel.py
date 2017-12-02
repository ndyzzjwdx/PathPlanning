# -*- coding: UTF-8 -*- 

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse

import ppfunction as ppf

# 原始数据各地坐标点维度
X_MAX = 548 
Y_MAX = 421

# 飞行器出发时间点及区域行驶时长
# T_START (hh) / T_STEP (mm) / T_END (hh)
T_START = 3
T_STEP = 2
T_END = 21

# 警告（坠毁）风速
WARNING_SPEED = 15

# 模型总数
MODELS = 10

# 存储数据表获取（空输入数据表获取）
def get_input_net(times = 1):
    width = MODELS * times
    length = (T_END - T_START + 1) * X_MAX * Y_MAX
    table = np.zeros((length, width))
    return table

# site(x,y,t) --> p 的映射规则
# 需要坐标维度X_MAX/Y_MAX
def get_index(x, y, t):
    p = (t - T_START) * (X_MAX * Y_MAX) + (x - 1) * Y_MAX + (y - 1)
    return p

# 模型合并规则 (simple: 取最大值)
# 需要风速预警值WARNING_SPEED
def get_warning_level(wind_list):
    wind = max(wind_list) / WARNING_SPEED
    return wind

# 获取方向向量
def get_direction(ori, goal):
    direction = [1,1]
    length = [0,0]
    for i in range(2):
        if ori[i] > goal[i]:
            direction[i] = -1
        length[i] = abs(goal[i] - ori[i]) + 1

    return direction, length

# 获取每一步的时刻
# 需要出发时间点(mm)及行驶步长T_STEP
def get_step_time(step, start_time = T_START*60):
    time = start_time + step * T_STEP
    hour = time // 60
    min = time - hour * 60
    return hour, min

# 起飞延时策略
def delay_start(wind_table, ori, direction):
    first_x = [ori[0] + direction[0], ori[1]]
    first_y = [ori[0], ori[1] + direction[1]]
    
    hour = T_START - 1
    flag = True

    while flag:
        hour = hour + 1
        if hour >= 21:
            break
        p0 = get_index(ori[0], ori[1], hour)
        p1 = get_index(first_x[0], first_x[1], hour)
        p2 = get_index(first_y[0], first_y[1], hour)

        if wind_table[p0] < 1:
            for i in [p1, p2]:
                if wind_table[i] < 1:
                    flag = False

    plan_pre = []
    steps = (hour - T_START) * 60 // 2
    for i in range(steps):
        plan_pre.append(ori)

    return  hour, plan_pre

# 风速网络生成策略
def create_wind_net(wind_table, ori, goal, time = T_START*60, offset = 0):
    direction, length = get_direction(ori, goal)
    wind_net = np.zeros(length)
    site = ori
    step = 0

    for i in range(ori[0], goal[0] + direction[0], direction[0]):
        for j in range(ori[1], goal[1] + direction[1], direction[1]):
            xmove = abs(i - ori[0])
            ymove = abs(j - ori[1])
            
            step = xmove + ymove + offset
            h, m = get_step_time(step, time)
            p = get_index(i, j, h)
            try:
            	wind_net[xmove, ymove] = wind_table[p]
            except IndexError:
                wind_net[xmove, ymove] = 1

    return wind_net

# 风险网络转化策略
def turn_SUI(wind_table):
    space = wind_table.shape
    network = np.zeros(space)
    bifurcation = []

    xm = space[0] - 1
    ym = space[1] - 1
    network[0,0] = 1
    for i in range(xm):
        for j in range(ym):
            # 读取当前点流量/下一步坐标点风速
            flow = network[i,j]
            wx = wind_net[i+1,j]
            wy = wind_net[i,j+1]

            # 根据下一步坐标点风速分配流量
            if wx > 1:
                if wy > 1:
                    param_x = 0
                    param_y = 0
                    bifurcation.append([i,j,flow])
                else:
                    param_x = 0
                    param_y = 1
            else:
                if wy > 1:
                    param_x = 1
                    param_y = 0
                else:
                    param_x = ppf.div(wy, wx)
                    param_y = ppf.div(wx, wy)
            # 记录流量转移
            network[i+1, j] = network[i+1, j] + param_x * flow
            network[i, j+1] = network[i, j+1] + param_y * flow

    for i in range(1,space[0]):
        j = space[1] - 1
        network[i, j] = network[i, j] + network[i-1, j]
    for j in range(1,space[1]):
        i = space[0] - 1
        network[i, j] = network[i, j] + network[i, j-1]

    rate_conn = network[xm, ym]

    return network, bifurcation, rate_conn

# 路径选择策略
# output: 0: x 轴移动/ 1: y 轴移动
def path_choose(network):
    plan_rel = []
    space = network.shape
    x = 0
    y = 0
    xlen = space[0] - 1
    ylen = space[1] - 1

    for i in range(xlen + ylen):
        if x == xlen:
            y = y + 1
            plan_rel.append(1)
        else:
            if y == ylen:
                x = x + 1
                plan_rel.append(0)
            else:        
                a = network[x+1, y] * (xlen - x)
                b = network[x, y+1] * (ylen - y)
                if a >= b:
                    x = x + 1
                    plan_rel.append(0)
                else:
                    y = y + 1
                    plan_rel.append(1)

    return plan_rel

# 路径坐标绝对化
def planning(network, ori, goal, plan_pre = []):
    plan = plan_pre + [ori]
    plan_rel = path_choose(network)
    
    direction, length = get_direction(ori, goal)
    site = ori
    for i in plan_rel:
        site = [site[0] + (1 - i) * direction[0], site[1] + i * direction[1]]
        plan.append(site)
    return plan

# 评测路径规划结果（代价函数）
# 结果以分数显示
def contrast(plan, winds_table_real, time_end):
    crash = [0,0]
    crash_flag = False
    overtime_flag = False    
    score = len(plan) * 2

    for i in range(len(plan)):
        hour, min = get_step_time(i)
        p = get_index(plan[i][0], plan[i][1], hour)
        wind = float(winds_table_real[p])
        if wind >= 1:
            crash = plan[i]
            crash_flag = True
            break
    if crash_flag:
        score = 24 * 60 
    if time_end > T_END:
        overtime_flag = True
        score = 24 * 60

    return score, crash_flag, overtime_flag, crash
 # 路径规划结果预估（test）
def contrast_test(plan, net_conn, time_end):
    crash_flag = False
    overtime_flag = False    
    score = len(plan) * 2
    
    if net_conn == 0:
        crash_flag = True
        score = 24 * 60
    if time_end > T_END:
        overtime_flag = True
        score = 24 * 60

    return score, crash_flag, overtime_flag



# 导出格式规范化
# 导出格式为['city', 'date', 'time'(hh:mm), 'xid', 'yid']
def get_out_path(plan, city, day):
    plan_df = pd.DataFrame(plan, columns = ['xid', 'yid'])
    plan_df['city'] = city
    plan_df['date'] = day
    plan_df['time'] = [t_str(elem) for elem in plan_df.index]
    output = plan_df.reindex(columns = ['city', 'date', 'time', 'xid', 'yid'])
    hour_end, min_end = get_step_time(plan_df.index[-1])
    return output, hour_end
# 时间格式化
def t_str(step):
    hour, min = get_step_time(step)
    return str(hour) + ':' + str(min).zfill(2)

# 导出反馈文件
def get_report(city, day, time_start, time_end, net_conn, crash_flag, overtime_flag, score):
    return [city, day, time_start, time_end, net_conn, crash_flag, overtime_flag, score]
def get_report_title():
    return ['city', 'day', 'time_start', 'time_end', 'net_conn', 'crash', 'overtime', 'score']
    














###### 待考虑策略 ######

# 联通风险网络生成策略
def create_SUI_conn(wind_table, wind_net, ori, goal, time = T_START*60, offset = 0):
    space = wind_net.shape
    network = wind_net
    time_net = np.zeros(space)
    wait_net = np.zeros(space)
    steps = space[0] + space[1] - 2
    step = 0

    while step < steps:
        step = step + 1
        flag = True
        dot = []
        for i in range(step + 1):
            j = step - i
            dot.append(wind_net[i,j])
        for k in dot:
            if k < 1:
                flag = False
        if flag:
            hour, min = get_step_time(time, step)
            step_stop = (60 - min) // 2 - 1
            step = step - 1
            network
            # 根据步数step行dot计算重置范围和重置起始时间time
            # 范围内time_net加1，计offset加1
            if step <= steps//2
                step = 0
                time = 
            flag = False
        else:
            break

    return wind_net, time_net

# 歧点消除策略
def remove_branch(table, ori, goal):
    branch = list()
    time = T_START
    step = 0

    direction, length = get_direction(ori, goal)
    [x, y] = ori
    for x != goal[0]:
        for y != goal[1]:
            
