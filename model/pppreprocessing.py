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
# T_START (hh) / T_STEP (mm)
T_START = 9
T_STEP = 2

# 警告（坠毁）风速
WARNING_SPEED = 20

# 计算行进至坐标点的时间映射规则
# 需要出发时间点T_START及行驶步长T_STEP
def time_validity(ori, goal):
    t_end = T_START + (abs(goal[0] - ori[0]) + abs(goal[1] - ori[1])) * T_STEP // 60     
    return t_end

# 获取每一步的时刻
# 需要出发时间点T_START及行驶步长T_STEP
def get_step_time(step):
    temp = step * T_STEP // 60
    hh = T_START + temp
    mm = step * T_STEP - temp * 60
    return str(hh) + ':' + str(mm)

# site(x,y,t) --> p 的映射规则
# 需要坐标维度X_MAX/Y_MAX
def get_index(x, y, t):
    p = (t - 1) * (X_MAX *Y_MAX) + (x - 1) * Y_MAX + (y - 1)
    return p

# 预测数据--模型合并规则 (simple: 取最大值)
# 需要坐标维度X_MAX/Y_MAX 和 风速预警值WARNING_SPEED
def weather_forecast(data):
    table = pd.DataFrame(columns = ['wind'])
    for t in data.hour.unique():
        for x in range(X_MAX):
            for y in range(Y_MAX):
                wind = data.loc[data.hour == t].loc[data.xid == x].loc[data.yid == y].wind.max() / WARNING_SPEED
                p = get_index(x, y, t)
                table.loc[p] = [wind]
    return table

# 真实风速读取
# 需要坐标维度X_MAX/Y_MAX 和 风速预警值WARNING_SPEED
def weather_real(data):
    table = pd.DataFrame(columns = ['wind'])
    for t in data_p.hour.unique():
        for x in range(X_MAX):
            for y in range(Y_MAX):
                wind = data.loc[data.hour == t].loc[data.xid == x].loc[data.yid == y].wind / WARNING_SPEED
                p = get_index(x, y, t)
                table.loc[p] = [wind]
    return table

# 载入起终点坐标数据
# citys_name = ['cid', 'xid', 'yid'], 'cid' == 0 为出发城市坐标点    
def load_cities(dirname, filename):
    data = ppf.load_data(dirname, filename)
    sites = pd.DataFrame(columns = ['xid', 'yid'])
    for i in range(len(data)):
        sites.loc[data[i][['cid']]] = data[i][['xid', 'yid']]

    cities = sites.sort_index(by = ['cid'])
    city_index = sites.cid.unique()
    return city_index, cities
    
# 载入风速数据
# winds_name = ['xid', 'yid', 'date_id', 'hour', 'realization', 'wind'(float)]
def load_winds(dirname, filename, cities):
    data = ppf.load_data(dirname, filename)
    wind_table = dict()
    
    # 按日期分组,压缩生成预测风险表
    date_index = data.date_id.unique()
    for day in date_index:
        data_p = data.loc[data.date_id == day][['xid','yid','hour','realization','wind']]
        table = weather_forecast(data_p)
        
        winds_area = cut_winds_area(table, cities)
        wind_table.setdefault(day, winds_area)
        
    wind_table.setdefault('date_index', date_index)    
    
    return wind_table

# 载入真实风速数据
# winds_real_name = ['xid', 'yid', 'date_id', 'hour', 'wind'(float)]
def load_winds_real(dirname, filename, cities):
    data = ppf.load_data(dirname, filename)
    wind_table_real = dict()

    # 按日期分组, 压缩生成真实风险表
    date_index = data.date_id.unique()
    for day in date_index:
        data_p = data.loc[data.date_id == day][['xid','yid','hour','wind']]
        table = weather_real(data_p)
        
        winds_area = cut_winds_area(table, cities)
        wind_table_real.setdefault(day, winds_area)
        
    wind_table_real.setdefault('date_index', date_index)    
    
    return wind_table_real

# 路径区域数据提取(分组)
def cut_winds_area(winds_table, cities):
    winds_area = dict()

    city_index = cities.cid.unique()[1:len(cities)]
    for city in city_index:
        area = cities.loc[[0,city]]
        x_ind = range(area.min()[0], area.max()[0])
        y_ind = range(area.min()[1], area.max()[1])
        t_ind = winds_table.hour.unique()

        area = winds_table.loc[[get_index(x, y, t) for x in x_ind for y in y_ind for t in t_ind]]
        winds_area.setdefault(city,area)
        
    return winds_area

# 导出行规范化
# 导出格式为['city', 'date', 'time'(hh:mm), 'xid', 'yid']
def get_out_path(plan, city, day):
    plan['city'] = city
    plan['date'] = date
    plan['time'] =  get_step_time(plan.index)

    output = plan.reindex(columns = ['city', 'date', 'time', 'xid', 'yid'])
    return output

# 导出为规范输出数据（data ==> txt）
# 导出数据格式固定，无需表头
def save_data(data, dirname, filename):
    data_list = list()
    data_list = data_list.append(ppf.list_to_str([row for row in data,get_values()]))
    data_str = ppf.list_to_str([elem for elem in data_list], '\n')
    
    flag = ppf.save_file_oss(data_str, dirname, filename)
    
    return flag

