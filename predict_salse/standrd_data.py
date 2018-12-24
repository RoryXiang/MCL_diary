# -*- coding:utf-8 -*-
# Date   : （Sta Dec 22 11:42:05 2018 +0800）
# Author : Rory Xiang

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from liner_regression.regression import stand_regres, lwlr


data_ = pd.read_excel("../predict_salse/salse_data.xlsx")
data = data_[["vacationd_time", "salse_time", "old", "salse"]]
print(np.array(data))


def get_origin_x_y_arr():
    xarr_ = data[["vacationd_time", "salse_time", "old"]]
    yarr_ = data["salse"]
    yarr = yarr_.tolist()
    xarr = []
    for row in xarr_.itertuples():
        xarr.append([row[1], row[2], row[3]])
    yarr = np.mat(yarr, dtype="float64")
    xarr = np.mat(xarr, dtype="float64")
    return xarr, yarr


def get_stanred_x_y_arr():
    data_all = np.array(data)  # 将DataFrame数据转换成array
    standred = StandardScaler()
    standred_data = standred.fit_transform(data_all)
    stand_x = []
    stand_y = []
    for one in standred_data:
        stand_x.append(one[:-1])
        stand_y.append(one[-1])

    return stand_x, stand_y


if __name__ == '__main__':
    stand_x, stand_y = get_stanred_x_y_arr()
    # ----------------------
    print("standx------------------")
    print(stand_x)
    print("standy------------------")
    print(stand_y)
    # stand_y = np.mat(stand_y)
    # ws = stand_regres(xarr, yarr)
    ws = stand_regres(stand_x, stand_y)
    print("ws----------------------")
    # print(ws)


if __name__ == '__main__':
    a = [30, 38, 166806]
    a = np.mat(a)
    # b = stand_data(a)
    # print(b)
    c = np.sum(a * ws)
    # c = lwlr(a, stand_x, stand_y, k=1)
    print("2019 winter will salse : ", c)