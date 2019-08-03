#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-03 15:47:07
# @Author  : RoryXiang (xiangshangping19901121@gmail.com)
# @Link    : ${link}
# @Version : $Id$


# import numpy as np
# from  sklearn import datasets
# from matplotlib import pyplot as plt
# data = datasets.make_blobs(20, n_features=2, centers=4, random_state=2)
# print(data)
# plt.scatter(data[0][:,0], data[0][:, 1], c=data[1], s=60)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# generate some random cluster data
X, y = make_blobs(random_state=170, n_samples=600, centers = 5)
rng = np.random.RandomState(74)
# transform the data to be stretched
transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)
# plot
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()


class ClassName(object):
    """docstring for ClassName"""
    def __init__(self):
        super().__init__()
        pass

    def culculate_distance(a, b):
        
        pass