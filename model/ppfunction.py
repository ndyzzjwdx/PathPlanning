# -*- coding: UTF-8 -*- 

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse

import tensorflow as tf
import tflearn
import tarfile

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tensorflow.python.lib.io import file_io

from six.moves import urllib

# 读取阿里云平台OSS路径
# FLAGS = None
# def get_fpath():
#    parser = argparse.ArgumentParser()
#    #获得buckets路径
#    parser.add_argument('--buckets', type=str, default='',
#                        help='input data path')
#    #获得checkpoint路径
#    parser.add_argument('--checkpointDir', type=str, default='',
#                        help='output model path')
#    FLAGS, _ = parser.parse_known_args()
#    #获得oss项目路径
#    dirname = os.path.join(FLAGS.buckets, "")
#
#    return dirname
def get_fpath():
    return 'oss://pathplaning/'

# 读取本地路径
def get_fpath_local():
    return  'C:/0/FutureWeather/'

# 阿里云平台OSS文件导入
def load_file_oss(dirname, loadname):
    fpath = os.path.join(dirname, loadname)
    object = file_io.read_file_to_string(fpath)
    return object

# 阿里云平台OSS导出
def save_file_oss(data, dirname, savename):
    try:
        tf.gfile.FastGFile(dirname + savename, 'wb').write(data)
    except Exception:
        return False
    return True

# 本地文件导入
def load_file_local(dirname, filename):
    if os.path.exists(dirname):
        fp = open(os.path.join(dirname, filename))
        return fp
    else:
        print('cannot open the file {}'.format(filename))
        return 0

# 本地文件导出
def save_file_local(data, dirname, savename):
    fh = open(os.path.join(dirname, savename), 'w')
    fh.write(data)
    fh.close()
    return 0

# 获取目录下文件名
def get_filename(dirname, dirname_middle, suffix):   
    L = list()
    for root, dirs, files in os.walk(os.path.join(dirname, dirname_middle)):  
        for file in files:  
            if os.path.splitext(file)[1] == suffix:  
                L.append(file)  
    return L 

# 单行字符串转化为列表
def str_to_list(str, sign = ',', type = str):
    data_list = [type(elem) for elem in str.split(sign)]    
    return data_list

# 单行列表转化为字符串
def list_to_str(data, sign = ','):
    data_str =  sign.join([str(e) for e in data])
    return data_str

# 列表转化为csv字符串
def list_to_csv(data, data_title = ''):
    data_list = list()
    for row in data:
        item = ','.join([str(e) for e in row])
        data_list.append(item)
    data_str = data_title + '\n'.join([elem for elem in data_list])

    return data_str

# csv字符串转化为字符列表
def csv_to_list(data_str, header = True):
    data_title = []
    data = data_str.split("\n")
    if header:
        data_title = data[0].split(',')
        data = data[1::]

    data_list = [elem.split(',') for elem in data]

    return data_list, data_title

# 取比例(含除0)
def div(a, b):
    if a == b == 0:
        return 1
    else:
        return a / (a + b)

# 自定义文件名命名规则
def get_name(city, day):
    name = 'c' + str(city) + 'd' + str(day);
    return name

# 导入数据（csv格式 ==> data）
# 以字符串列表形式导入
def load_data(dirname, filename, header = True):    
    object = load_file_oss(dirname, filename)        
    data, data_title = csv_to_list(object, header = header)
    return data, data_title

# 导入本地数据（csv格式 ==> data）
# 以字符串形式导入
def load_csv_local(dirname, filename, header = True):
    fp = load_file_local(dirname, filename)
    data_title = []
    data_str = fp.readlines()
    if header:
        data_title = data_str[0].replace('\n','').split(',')
        data_str = data_str[1::]
    data_list = [elem.replace('\n','').split(',') for elem in data_str]

    fp.close()

    return data_list, data_title

# 导入本地数据（txt格式 ==> data）
# 默认以float类型转码导入
def load_data_local(dirname, filename, type = float):
    fp = load_file_local(dirname, filename)
    data_str = fp.readline()
    data = str_to_list(data_str, sign = ',', type = type)
    fp.close()

    return data

