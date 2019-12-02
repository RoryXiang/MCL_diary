#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/19 10:36
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
# style.use("ggplot")


def line():
    x = [5, 2, 7]
    y = [2, 16, 4]
    x1 = [5, 8, 10]
    y1 = [12, 16, 6]
    plt.plot(x, y, color="g", linewidth=2, label="line one", linestyle="-")
    plt.plot(x1, y1, color="r", linewidth=2, label="line two", linestyle="--")
    plt.title("Image Title")  # define the title of the pic
    plt.ylabel("Y axis")  # define the name of y axis
    plt.xlabel("X axis")  # define the name of x axis
    plt.legend()  # no label without this sentence
    plt.show()


def bar_chart():
    """
    function bar and hist
    """
    # ------------------- style one -------------------------------------
    # plt.bar([0.25, 1.25, 2.25, 3.25, 4.25], [50, 40, 70, 80, 20], label="BMW",
    #         color='b', width=.5)
    # plt.bar([.75, 1.75, 2.75, 3.75, 4.75], [80, 20, 20, 50, 60], label="Audi",
    #         color='r', width=.5)
    # plt.legend()
    # plt.xlabel('Days')
    # plt.ylabel('Distance (kms)')
    # plt.title('Information')
    # plt.show()
    # --------------------------------------------------------------------

    # -----------------------style two -----------------------------------
    population_age = [22, 55, 62, 45, 21, 22, 34, 42, 42, 4, 2, 102, 95, 85, 55,
                      110, 120, 70, 65, 55, 111, 115, 80, 75, 65, 54, 44, 43,
                      42, 48]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.hist(population_age, bins, histtype='bar', color='b', rwidth=0.8)
    plt.xlabel('age groups')
    plt.ylabel('Number of people')
    plt.title('Histogram')
    plt.show()


def point():
    """
    function scatter
    """
    x = [1, 1.5, 2, 2.5, 3, 3.5, 3.6]
    y = [7.5, 8, 8.5, 9, 9.5, 10, 10.5]
    x1 = [8, 8.5, 9, 9.5, 10, 10.5, 11]
    y1 = [3, 3.5, 3.7, 4, 4.5, 5, 5.2]
    plt.scatter(x, y, label="heigh income low saving", color="r")
    plt.scatter(x1, y1, label="low income heigh saving", color="b")
    plt.xlabel("saving*100")
    plt.ylabel("income*100")
    plt.title("Scatter plot")
    plt.legend()
    plt.show()


def pie_chart():
    days = [1, 2, 3, 4, 5]
    sleeping = [7, 8, 6, 11, 7]
    eating = [2, 3, 4, 3, 2]
    working = [7, 8, 7, 2, 2]
    playing = [8, 5, 7, 8, 13]
    slices = [7, 2, 2, 13]
    activities = ['sleeping', 'eating', 'working', 'playing']
    cols = ['c', 'm', 'r', 'b']
    plt.pie(slices, labels=activities, colors=cols, startangle=90, shadow=False,
            explode=(0, 0.1, 0, 0), autopct="%1.1f%%")
    plt.title("Pie plot")
    plt.show()


def multp():
    """
    function subplot
    """
    def f(t):
        return np.exp(-t) * np.cos(2 * np.pi * t)

    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    plt.subplot(221)
    plt.plot(t1, f(t1), 'bo', t2, f(t2))
    plt.subplot(224)
    plt.plot(t2, np.cos(2*np.pi*t2))
    plt.show()


if __name__ == "__main__":
    # line()
    # bar_chart()
    # point()
    # pie_chart()
    multp()
