# -*- coding:utf-8 -*-
# Date   : （Sta Dec 22 18:46:05 2018 +0800）
# Author : Rory Xiang

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from liner_regression.regression import stand_regres, lwlr
from predict_salse.standrd_data import get_stanred_x_y_arr, get_origin_x_y_arr


def pridict_by_liner_regression(xarr, yarr, xtest):
    ws = stand_regres(xarr, yarr)
    c = np.sum(xtest * ws)
    return c



