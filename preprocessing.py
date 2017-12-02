#!python3
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

# 数据预处理
# test：True(测试数据)/False(真实数据)
def pp_data(dirname, dirname_save, dirname_middle, test = True):
    file_list = ppf.get_filename(dirname, dirname_middle, '.csv')
    for filename in file_list:
        table = ppm.get_input_net()
        file = ppf.load_file_local(dirname, dirname_middle + filename)

        for i in range(len(table)):
            line = file.readline().replace('"','').split(',')
            l = [int(e) for e in line[0:3]]
            if test:
                w = [float(e) for e in line[3:13]]
            else:
                w = float(line[3])
            p = ppm.get_index(l[0],l[1],l[2])
            wind = ppm.get_warning_level(w)
            table[p,0] = wind

        file.close()

        savename = dirname_middle + filename[0:len(filename)-4] + '.txt'
        data = ','.join([str(e) for e in table.ravel()])
        ppf.save_file_local(data, dirname_save, savename)
        print('Save ' + savename + ' done.')
    

if __name__ == '__main__':
    # 导入预处理数据
    dirname = ppf.get_fpath_local();
    input_dir = os.path.join(dirname, 'data/data/')
    output_dir = os.path.join(dirname, 'input/')    

    # 分别处理各项数据
    print('start to save files.')
    pp_data(input_dir, output_dir, 'train/')
    pp_data(input_dir, output_dir, 'real/', test = False)
    pp_data(input_dir, output_dir, 'test/')

