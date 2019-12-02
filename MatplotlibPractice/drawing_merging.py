#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/21 15:39
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


"""
双y轴绘图及合并
"""


def tw_fun():
    time = np.arange(10)
    temp = np.random.random(10) * 30
    swdown = np.random.random(10) * 100 - 10
    rn = np.random.random(10) * 100 - 10
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, swdown, "-", label="swdown")
    ax.plot(time, rn, '-', label='Rn')
    ax2 = ax.twinx()
    ax2.plot(time, temp, '-r', label='temp')
    ax.legend(loc=2)
    ax.grid()
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
    ax2.set_ylabel(r"Temperature ($^\circ$C)")
    ax2.set_ylim(0, 35)
    ax.set_ylim(-20, 100)
    ax2.legend(loc=0)
    plt.show()


"""
figure图的嵌套
"""

def figure_draw():
    # 定义figure
    fig = plt.figure()

    # 定义数据
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 3, 4, 2, 5, 8, 6]

    # figure的百分比, 从figure 10%的位置开始绘制, 宽高是figure的80%
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    # 获得绘制的句柄
    ax1 = fig.add_axes([left, bottom, width, height])
    # 绘制点(x,y)
    ax1.plot(x, y, 'r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('test')

    # 嵌套方法一
    # figure的百分比, 从figure 10%的位置开始绘制, 宽高是figure的80%
    left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
    # 获得绘制的句柄
    ax2 = fig.add_axes([left, bottom, width, height])
    # 绘制点(x,y)
    ax2.plot(x, y, 'r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('part1')

    # # 嵌套方法二
    plt.axes([bottom, left, width, height])
    plt.plot(x, y, 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('part2')

    plt.show()


if __name__ == "__main__":
    # tw_fun()
    figure_draw()