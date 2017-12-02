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

# 当前警告风速
WARN_SPEED = 15

# 读取统计值
def statistics(data, norm):
    tp = list()
    fp = list()
    fn = list()
    tn = list()

    for row in data:
        if row[0] >= norm:
            if row[1] > norm:
                tp.append(row)
            else:
                fp.append(row)
        else:
            if row[1] > norm:
                fn.append(row)
            else:
                tn.append(row)

    return tp, fn, fp, tn    

# 计算模型评估参数
def compute(tp, fn, fp, fn):
    d11 = len(tp)
    d01 = len(fn)
    d10 = len(fp)
    d00 = len(tn)

    presion = ppf.div(d11, d10) 
    recall = ppf.div(d11, d01)
    F = 2 * recall * ppf.div(presion, recall)       

    TPR = recall
    FPR = ppf.div(d10, d00)
    ROC = (FPR, TPR)

    return presion, recall, F, TPR, FPR

# 本地文件分解
def cut_data(fp_train, fp_real):
    data_str_train = fp_train.readline()
    data_str_real = fp_real.readline()

    data_train = ppf.str_to_list(data_str_train, sign = ',', type = float)
    data_real = ppf.str_to_list(data_str_real, sign = ',', type = float)

    data = np.mat([data_train, data_real]).transpose().tolist()

    fp_train.close()
    fp_real.close()

    return data

# 全文件读取/合并
def get_data(dirname_root, dirname_train, dirname_real):
    file_list = ppf.get_filename(dirname_root, dirname_train, '.txt')

    data = list()    
    for i in range(len(file_list)):
        fp_train = ppf.load_file_local(dirname_root, dirname_train + file_list[i])
        fp_real = ppf.load_file_local(dirname_root, dirname_real + file_list[i])
        data = data + cut_data(fp_train, fp_real).tolist()    
        
    return data

if __name__ == '__main__':
    # 设置I/O目录
    dirname = ppf.get_fpath_local();
    input_dir = os.path.join(dirname, 'input/')
    output_dir = os.path.join(dirname, 'output/')    
    
    # 导入训练数据
    print('Loading data...')
    data = get_data(input_dir, 'train/', 'real/') 
    
    # 按指标读取评价标准
    print('Computing data...')
    ans = np.zeros((30,6))
    for i in range(6,36):
        tp, fn, fp, tn = statistics(data, i/WARN_SPEED)
        ans[i-6,0] = i
        ans[i-6,1:6] = compute(tp, fn, fp, fn)

    # 保存评价
    print('Save answer...')
    data_title = 'wind,presion,recall,F,TPR,FPR\n'
    data_str = ppf.list_to_csv(table, data_title)
    save_data(data_str, output_dir, 'assessment.csv')
    print('Save report done.') 
