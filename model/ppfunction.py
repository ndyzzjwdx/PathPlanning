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

# 单行字符串转化为数字列表
def str_to_list(str, sign = ','):
    data_str = str.split(sign)
    data = list()
    for elem in data_str:
        try:
            data.append(int(elem))
        except Exception:
            try:
                data.append(float(elem))
            except Exception:
                data = []
                break
    
    return data

# 单行数字列表转化为字符串
def list_to_str(data, sign = ','):
    data_str =  sign.join([str(e) for e in data])
    return data_str

# 自定义文件名命名规则
def get_name(city, day):
    name = 'c' + str(day) + 'd' + str(city);
    return name

# 导入原始数据（csv ==> data）
# 原始数据由列名和数据组成
def load_data(dirname, filename):    
    object = load_file_oss(dirname, filename)        
    data = object.split("\n")
    data_title = data[0]
    data_str = data[1::]

    data_list = list()
    for row in data:
        data_list.append(str_to_list(row))
    data_table = pd.DataFrame(data, columns = data_name)

    return data_table

# 保存处理数据（data ==> txt）
# 处理数据为临时保留数据，行向量形式，'，'连接
# object[0:1]存储行列参数
def save_matrix(data, dirname, filename):
    data_size = list(np.shape(data))
    data_vector = np.ravel(data).tolist()
    
    data_str = list_to_str(data_size + data_vector)
    flag = save_file_oss(data_str, dirname, filename + '.txt')

    return flag


# 导入处理数据（txt ==> data）
# 处理数据为临时保留数据，行向量形式，'，'连接
# object[0:1]存储行列参数
def load_matrix(dirname, filename):
    object = load_file_oss(dirname, filename)
    data = str_to_list(object)
    data_size = data[0:2]
    data_vector = data[2::]

    data_matrix = np.reshape(data_vector,data_size)
    
    return data_matrix

