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

# 导入城市坐标
def load_cities(dirname, filename):
    cities, cities_title = ppf.load_csv_local(dirname, 'data/CityData.csv')
    ind = dict(zip(cities_title, range(len(cities_title))))
    c = ind['cid']
    x = ind['xid']
    y = ind['yid']

    data = [[int(row[x]),int(row[y])] for row in cities]
    columns = ['xid', 'yid']
    index = [int(row[c]) for row in cities]

    city_df = pd.DataFrame(data, columns = columns, index = index).sort_index()    

    return city_df

if __name__ == '__main__':
    # 导入数据目录
    dirname = ppf.get_fpath_local()
    input_dir = os.path.join(dirname, 'input/')
    output_dir = os.path.join(dirname, 'output/')    

    print('Loading cities...')
    # 导入城市坐标
    cities = load_cities(dirname, 'data/CityData.csv')
    # 导入规划数据
    plan_all = pd.DataFrame()
    score_all = 0
    report = []

    file_list = ppf.get_filename(input_dir, 'test/', '.txt')
    city_list = cities.index[1::].tolist()
    for file in file_list:
        print('Loading data of ' + file + '...')
        wind_train = ppf.load_data_local(input_dir, 'test/' + file)
        
        day = int(file[5:len(file)-4]);
        for city in city_list:
            # 使用模型计算路径
            # 参数设定
            ori = cities.loc[0].tolist()
            goal = cities.loc[city].tolist()
            direction, length = ppm.get_direction(ori, goal)
            name = ppf.get_name(city, day)
            print('Start disposing file ' + name + '...')
            
            # 提取风速网络
            time_start, plan_pre = ppm.delay_start(wind_train, ori, direction)
            wind_net = ppm.create_wind_net(wind_train, ori, goal, time = time_start * 60)
            # 计算风险网络
            network, bifurcation, net_conn = ppm.turn_SUI(wind_net)
            if len(bifurcation) != 0:
                network, bifurcation, net_conn = ppm.remove_bifurcation(wind_net, bifurcation)
            print('Net connect = ' + str(net_conn))

            # 路径规划
            plan = ppm.planning(network, ori, goal, plan_pre = plan_pre)
            # 路径格式规范化及评价
            output, time_end = ppm.get_out_path(plan, city, day)
            score, crash_flag, overtime_flag = ppm.contrast_test(plan, net_conn, time_end)
            if crash_flag:
                print('The plane maybe crash.')
            else:
                if overtime_flag:
                    print('The plane overtime.')
                else:
                    print('score = ' + str(score))
            
            # 保存路径
            if crash_flag == False:
                if overtime_flag == False:
                    plan_all = plan_all.append(output)
            score_all = score_all + score
            report.append(ppm.get_report(city, day, time_start, time_end, net_conn, crash_flag, overtime_flag, score))

            ppf.save_file_local(output.to_csv(header = False, index = False), output_dir, 'train/plan_' + name +  '.csv')
            print('Save ' + name + ' done.')

    # 保存提交文件
    ppf.save_file_local(plan_all.to_csv(header = False, index = False), output_dir, 'pathplanning_test.csv')
    print('Save pathplanning done.')
    print('Score = ' + str(score_all))
    # 保存报告文件
    out_report = pd.DataFrame(report, columns = ppm.get_report_title()).to_csv(index = False)
    ppf.save_file_local(out_report, output_dir, 'report_test.csv')
    print('Save report done.')

