"""蚂蚁算法解决TSP问题
    核心需要理解蚂蚁在行经中下一个point选择概率函数；以及影响概率函数的信息素的更新
"""

import numpy as np
import sys
import random
import math
import tkinter
from functools import reduce
import copy

ALPHA = 1.0  # 信息启发因子
BATA = 1.0  # 期望值启发因子
RHO = 0.5  # 信息素挥发因素
Q = 1  # 信息素总量
city_num = 50  # 城市总数
ant_num = 50  # 蚁群数量

# 初始化城市位置信息
city_positions = np.random.randint(0, 700, size=(city_num, 2))
# 初始化距离图谱
distance_graph = np.zeros((city_num, city_num))
# 初始化信息素
pheromone_graph = np.ones((city_num, city_num))


# 蚂蚁类=========================================


class Ant(object):
    """docstring for ClassName"""

    def __init__(self, ID):
        self.ID = ID
        self.__init_data()  # 初始化蚂蚁，选择出生点

    def __init_data(self):
        self.path = []  # 蚂蚁的路经
        self.callable_city = np.ones((city_num,), dtype=np.bool)  # 城市可访问状态
        self.total_distance = 0.0  # 蚂蚁的路径总距离
        self.current_city = random.randint(0, city_num - 1)  # 选择初始化城市
        self.path.append(self.current_city)
        self.callable_city[self.callable_city] = False
        self.move_count = 1

    def __choice_next_city(self):
        # 初始化转移概率矩阵（从当前城市到每一个城市的概率）
        cities_prob = np.zeros(shape=(city_num,))
        # 用概率函数计算概率矩阵
        for next_city in range(city_num):
            if self.callable_city[next_city]:
                try:
                    cities_prob[next_city] = np.power(
                        pheromone_graph[self.current_city][next_city], ALPHA) * np.power(1.0 / distance_graph[self.current_city][next_city], BATA)
