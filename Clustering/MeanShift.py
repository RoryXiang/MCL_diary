# coding=utf-8

"""
meanshift聚类算法
核心思想：
寻找核密度极值点并作为簇的质心，然后根据最近邻原则将样本点赋予质心
"""
from collections import defaultdict
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.datasets import make_blobs
import time


class MeanShift:
    def __init__(self, epsilon=0.001, band_width=2, min_fre=5):
        self.epsilon = epsilon
        self.band_width = band_width
        self.min_fre = min_fre  # 可以作为起始质心的球体内最少的样本数目
        self.radius2 = self.band_width ** 2  # 高维球体半径的平方

        self.N = None
        self.centers = []  # 存放所有的族心
        self.center_score = []  # 每个簇心球内包含的的个数

    def init_param(self, data):
        # 初始化参数
        self.N = data.shape[0]  # 一共多少个点
        self.labels = -1 * np.ones(self.N)  # 每个point所属的簇心
        return

    def initial_centers(self, data):
        # 获取可以作为起始簇心的点
        initial_centers = []
        centers_fre = defaultdict(int)
        for sample in data:
            # 将数据粗粒化，以防止非常近的样本点都作为起始质心
            seed = tuple(np.round(sample / self.band_width))
            centers_fre[seed] += 1
        for seed, fre in centers_fre.items():
            if fre >= self.min_fre:
                initial_centers.append(np.array(seed) * self.band_width)
        if not initial_centers:
            raise ValueError('the bin size and min_fre are not proper')
        # print(len(seed_list), seed_list)
        return initial_centers

    def euclidean_dis2(self, center, sample):
        # 计算均值点到每个样本点的欧式距离（平方）
        delta = center - sample
        return delta @ delta

    def gaussian_kel(self, dis2):
        # 计算高斯核, 其实就是求每个元素相对中心的权重
        left = 1.0 / (self.band_width * (2 * math.pi))
        right = math.exp(-(dis2 ** 2) / self.band_width ** 2)
        return left * right

    def shift_center(self, current_center, data, tmp_center_score):
        # ======== 核心函数 =============
        # 计算下一个漂移的坐标
        # ===============================
        denominator = 0  # 分母
        numerator = np.zeros_like(current_center)  # 分子, 一维数组形式
        for ind, sample in enumerate(data):
            dis2 = self.euclidean_dis2(current_center, sample)
            if dis2 <= self.radius2:
                tmp_center_score += 1
                wight = self.gaussian_kel(dis2)
                denominator += wight
                numerator += wight * sample  # 将球内每个点乘以权重相加
        return numerator / denominator

    def classify(self, data):
        # 根据最近邻将数据分类到最近的簇中
        center_arr = np.array(self.centers)
        for i in range(self.N):
            delta = center_arr - data[i]
            dis2 = np.sum(delta * delta, axis=1)
            self.labels[i] = np.argmin(dis2)
        return

    def fit(self, data):
        """训练主函数"""
        self.init_param(data)
        initial_centers = self.initial_centers(data)
        # for cluster_center in seed_list:
        #     plt.plot(cluster_center[0], cluster_center[1], 'o',
        #              markerfacecolor="m", markeredgecolor='k', markersize=14)
        # plt.pause(0.05)
        for center in initial_centers:
            tmp_center_score = 0
            # 进行一次独立的均值漂移
            while True:
                next_center = self.shift_center(
                    center, data, tmp_center_score)
                delta_dis = np.linalg.norm(
                    next_center - center, 2)  # 求向量的范数（长度）
                if delta_dis < self.epsilon:
                    break
                center = next_center
            # 若该次漂移结束后，最终的质心与已存在的质心距离小于带宽，则合并
            for i in range(len(self.centers)):
                if np.linalg.norm(center - self.centers[i], 2) < self.band_width:
                    # 如果两个簇心距离小于带宽，选择球内点多的作为簇心
                    if tmp_center_score > self.center_score[i]:
                        self.centers[i] = center
                        self.center_score[i] = tmp_center_score
                    break
            else:
                self.centers.append(center)
                self.center_score.append(tmp_center_score)
        self.classify(data)
        return


def visualize(data, labels, centers):
    """画图函数"""
    color = 'bgrymkc'
    unique_label = np.unique(labels)
    print(unique_label)
    for col, label in zip(cycle(color), unique_label):
        partial_data = data[np.where(labels == label)]
        plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col)
    for cluster_center in centers:
        plt.plot(cluster_center[0], cluster_center[1], 'o',
                 markerfacecolor=col, markeredgecolor='k', markersize=14)
    plt.show()
    return


if __name__ == '__main__':
    # 生成数据点
    # ==================== 随机现状的数据 =========================
    # data, _ = make_blobs(n_samples=500, centers=7,
    #                      cluster_std=1.5, random_state=10)

    # ==================== 特定形状的数据 =========================
    X, y = make_blobs(random_state=170, n_samples=500, centers = 5)
    rng = np.random.RandomState(74)
    # transform the data to be stretched
    transformation = rng.normal(size=(2, 2))
    data = np.dot(X, transformation)
    # ============================================================

    t1 = time.time()
    MS = MeanShift()
    # 聚类
    MS.fit(data)
    t2 = time.time()
    print("time  : ", t2 - t1)
    labels = MS.labels
    centers = MS.centers
    visualize(data, labels, centers)
