#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/21 19:35
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.externals import joblib

train_data = pd.read_csv("./data/titanic_train.csv")
test_data = pd.read_csv("./data/titanic_test.csv")

# print(test_data.info())
# print(train_data.describe())


# 特征选取
# 1、空数据处理
# Age 列中缺失值用中位数填充
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())

"""
线性回归
"""

# 选择简单可用的特征
predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]


def liner_reg():
    # 初始化线性回归算法
    alg = LinearRegression()

    # 样本均分成三分，交叉验证
    kf = KFold(n_splits=3, shuffle=False, random_state=1)

    predictions = []
    for train, test in kf.split(train_data):
        # train_predictors = (train_data[predictors].iloc[train, :])
        train_predictors = (train_data[predictors].iloc[train, :])
        train_target = train_data["Survived"].iloc[train]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(train_data[predictors].iloc[test, :])
        predictions.append(test_predictions)

    # joblib.dump(alg, "./l.pkl")  # 保存模型

    predictions = np.concatenate(predictions, axis=0)

    predictions[predictions > 0.5] = 1
    predictions[predictions < 0.5] = 0
    accuracy = sum(predictions == train_data["Survived"]) / len(predictions)
    print(accuracy)


def logstic_reg():
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    LogRegAlg = LogisticRegression(random_state=1)
    re = LogRegAlg.fit(train_data[predictors], train_data["Survived"])
    scores = model_selection.cross_val_score(
        LogRegAlg, train_data[predictors], train_data["Survived"], cv=3)
    print(scores.mean())


if __name__ == "__main__":
    logstic_reg()
