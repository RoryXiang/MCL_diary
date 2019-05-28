# coding=utf-8

import math
import numpy as np


class MeanShift(object):

    def __init__(self, bind_width=2, min_dis=0.00001):
        """
        :param bind_width： 球半径
        :return: min_dis： 最小漂移距离
        """
        self.bind_width = bind_width
        self.min_dis = min_dis
        self.centers = []

    def init_param(self, data):
        # 初始化参数
        self.N = data.shape[0]
        self.labels = -1 * np.ones(self.N)
        return

    def gaussian_weight(self, distance):
        """高斯核函数，当前球中每个点的权重
        :param distance： 当前点到球心的距离
        """
        left = 1.0 / (self.bind_width * (2 * math.pi))
        right = math.exp(-(distance ** 2) / self.bind_width ** 2)
        return left * right

    def euclidean_dis2(self, center, sample):
        # 计算均值点到每个样本点的欧式距离（平方）
        delta = center - sample
        return delta @ delta

    def shift_center(self, current_center, points):
        """计算当前球的均值球心
        :param current_center： 当前球心
        :param points： 所有点
        """
        all_wight = 0
        wights = []
        below_points = []
        for point in points:
            dis = self.euclidean_dis2(current_center, point)
            if dis <= self.bind_width ** 2:
                wight = self.gaussian_weight(dis)
                all_wight += wight
                wights.append(wight)
                below_points.append(point)
        next_center = np.array(below_points) * np.array(wights).T / all_wight
        return next_center

    def classify(self, points):
        center_arr = np.array(self.centers)
        for i in range(self.N):
            dealt = center_arr - points[i]
            dis_arr = np.sum(dealt * dealt, axis=1)  # 计算第i个点到每个center的距离
