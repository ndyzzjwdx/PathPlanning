# -*- coding: UTF-8 -*- 

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse

import ppfunction as ppf

# 生成风险网络
# type: 0 简单规避(无视歧点)/ 1 歧点消除/ 2 歧点停留
def create_SUI(wind_table, ori, goal, type = 0):
    network = 0
    return network

# 路径规划
def planning(network, ori, goal, type = 0):
    
    return 0

# 获取前进方向
# output: 0 原地/ 1 x向前进 / 2 y向前进
def get_next(wind_table, site):
    return 0
