# -*- coding: UTF-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse

sys.path.append(".")
import model.ppfunction as ppf
import model.ppmodel as ppm
import model.pppreprocessing as ppp

if __name__ == '__main__':
    # 导入数据目录
    dirname = get_fpath()
    
    # 导入城市坐标
    city_index, cities = ppp.load_cities(dirname, 'CityData.csv')

    # 导入风速数据
    winds_table = ppp.load_winds(dirname, 'ForecastDataforTesting.csv', cities)

    # 使用模型计算路径
    plan_all = pd.DataFrame()
    date_index = winds_table['date_index']
    ori = cities.loc[0]        
    for day in date_index:
        for city in city_index[1::]:
            # 提取待计算的风速数据
            winds_net = winds_table[day][city]
            goal = cities.loc[city]
            network = ppm.create_SUI(winds_net, ori, goal)
            plan = ppm.planning(network, ori, goal)
            
            # 计算并记录路径
            plan_all = plan_all.append(ppp.get_out_path(plan, city, day))

    # 输出路径文件
    ppp.save_data(plan_all, dirname, 'output/path_test.txt')    

