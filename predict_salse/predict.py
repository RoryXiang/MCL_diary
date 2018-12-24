# -*- coding:utf-8 -*-
# Date   : （Sta Dec 22 18:46:05 2018 +0800）
# Author : Rory Xiang

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from liner_regression.regression import stand_regres, lwlr
from predict_salse.standrd_data import get_stanred_x_y_arr, get_origin_x_y_arr


def predict_by_liner_regression(xarr, yarr, xtest):
    ws = stand_regres(xarr, yarr)
    c = np.sum(xtest * ws)
    return c


if __name__ == '__main__':
    xtest = [30, 38, 166806]
    xarr, yarr = get_origin_x_y_arr()
    stand_x, stnd_y = get_stanred_x_y_arr()
    salse = predict_by_liner_regression(xarr, yarr, xtest)
    stand_salse = predict_by_liner_regression(stand_x, stnd_y, xtest)
    print(salse, stand_salse)

