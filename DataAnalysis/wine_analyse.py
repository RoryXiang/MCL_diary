#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/21 16:06
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 颜色
color = sns.color_palette()
# 数据print精度
pd.set_option("precision", 3)

# load data
df = pd.read_csv("./data/winequality-red.csv", sep=';')
# print(df.head(5))
# print(df.info())

# 简单的数据统计
print("------------------ 简单的数据统计 ↓ ------------------")
print(df.describe())

# set plot style
plt.style.use("ggplot")

colnm = df.columns.tolist()  # 获取字段明，表头
print(colnm)


def subplot_draw1():
    fig = plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.subplot(2, 6, i + 1)  # subplots 表示分布绘制系列图
        sns.boxenplot(df[colnm[i]], orient="v", width=0.5, color=color[0])
        plt.ylabel(colnm[i], fontsize=12)

    # plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.tight_layout()  # 会自动调整子图参数，使之填充整个图像区域。避免重叠
    plt.show()


def subplot_draw2():
    fig = plt.figure(figsize=(10, 8))
    for i in range(12):
        plt.subplot(4, 3, i + 1)  # 设置子图的位置
        df[colnm[i]].hist(bins=100, color=color[0])
        plt.xlabel(colnm[i], fontsize=12)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    subplot_draw2()
