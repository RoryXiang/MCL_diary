# -*- coding:utf-8 -*-
# Date   : （Sta Dec 22 18:46:05 2018 +0800）
# Author : Rory Xiang

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from liner_regression.regression import stand_regres, lwlr
from predict_salse.standrd_data import (get_stanred_x_y_arr, get_origin_x_y_arr,
                                        get_everyday_origin_x_y_arr,
                                        get_everyday_stanred_x_y_arr)
from liner_regression.ridge_regression import rigre_regression, stage_wise


def predict_by_liner_regression(xarr, yarr, xtest):
    ws = stand_regres(xarr, yarr)
    c = np.sum(xtest * ws)
    return c


def predict_by_lwlr(xarr, yarr, xtest):
    c = lwlr(xtest, xarr, yarr)
    return np.sum(c)


def predict_by_rigre_regression(xarr, yarr, xtest):
    ws = stand_regres(xarr, yarr)  # 标准岭回归
    # ws = stage_wise(xarr, yarr)  # 前向逐步回归
    c = np.sum(xtest * ws)
    return c


if __name__ == '__main__':
    xtest = [30, 38, 166806]
    # xtest = [30, 214, 218803]
    xtest = np.mat(xtest)
    xarr, yarr = get_origin_x_y_arr()
    stand_x, stnd_y = get_stanred_x_y_arr()

    # xtest = [1, 381243]
    # xarr, yarr = get_everyday_origin_x_y_arr()
    # stand_x, stnd_y = get_everyday_stanred_x_y_arr()

    # liner -----------------------------------
    salse = predict_by_liner_regression(xarr, yarr, xtest)
    stand_salse = predict_by_liner_regression(stand_x, stnd_y, xtest)

    # regre------------------------------------
    rigre_salse = predict_by_rigre_regression(xarr, yarr, xtest)
    rigre_stand_salse = predict_by_rigre_regression(stand_x, stnd_y, xtest)

    # lwlr-------------------------------------
    lwlr_salse = predict_by_lwlr(stand_x, stnd_y, xtest)

    print("# liner------------------------------------")
    print(salse, stand_salse)
    print("# regre------------------------------------")
    print(rigre_salse, rigre_stand_salse)
    print("# lwlr------------------------------------")
    print(lwlr_salse)

    # rigre------------------------------------------------
    rigre_ws = rigre_regression(stand_x, stnd_y)
    rigre_salse = np.sum(rigre_ws*xtest)
    rigre_ws_ = rigre_regression(xarr, yarr)
    rigre_salse_ = np.sum(rigre_ws_ * xtest)
    print("# rigre------------------------------------")
    print(rigre_salse_, rigre_salse)

    # stage_wise---------------------------------------------
    stage_ws = stage_wise(stand_x, stnd_y)
    stage_ssalse = np.sum(stage_ws*xtest)
    print("# stage_wise------------------------------------")
    print(stage_ssalse)



