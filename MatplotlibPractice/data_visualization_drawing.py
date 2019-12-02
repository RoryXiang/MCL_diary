#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/19 16:34
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

"""
图表的基本元素
    图名
    x轴标签
    y轴标签
    图例
    x轴边界
    y轴边界
    x刻度
    y刻度
    x刻度标签
    y刻度标签
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 1 ==========================================
def fun():
    df = pd.DataFrame(np.random.rand(10, 2), columns=['A', 'B'])

    fig = df.plot(figsize=(8, 4))  # figsize：创建图表窗口，设置窗口大小


    plt.title('TITLETITLETITLE')  # 图名
    plt.xlabel('XXXXXX')  # x轴标签
    plt.ylabel('YYYYYY')  # y轴标签

    plt.legend(loc='upper left')  # 显示图例，loc表示位置

    plt.xlim([0,12])  # x轴边界
    plt.ylim([0,1.5])  # y轴边界

    plt.xticks(range(10))  # 设置x刻度
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])  # 设置y刻度

    fig.set_xticklabels("%.1f" % i for i in range(10))  # x轴刻度标签
    fig.set_yticklabels("%.2f" % i for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])  # y轴刻度标签

    plt.plot(np.random.rand(10))
    # plt.show()


"""
# 先建立子图然后填充图表
"""


def fun1():
   fig = plt.figure(figsize=(10, 6), facecolor='gray')

   ax1 = fig.add_subplot(2, 2, 1)
   plt.plot(np.random.rand(50).cumsum(), 'k--')
   plt.plot(np.random.randn(50).cumsum(), 'b--')

   ax2 = fig.add_subplot(2, 2, 2)
   ax2.hist(np.random.rand(50), alpha=0.5)

   ax4 = fig.add_subplot(2, 2, 4)
   df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
   ax4.plot(df2, alpha=0.5, linestyle='--', marker='.')
   # plt.show()


"""
多系列图绘制
"""


def fun2():
    """
    绘制系列图
    subplots，是否分别绘制系列（子图）
    layout：绘制子图矩阵，按顺序填充
    """
    ts = pd.Series(np.random.randn(1000).cumsum())
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                      columns=list('ABCD'))
    df = df.cumsum()
    print(df)
    df.plot(style='--.', alpha=0.4, grid=True, figsize=(20, 8),
            subplots=True,
            layout=(1, 4),
            sharex=False)
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.show()


"""
基本图绘制
参数含义：
    series的index为横坐标
    value为纵坐标
    kind → line,bar,barh...（折线图，柱状图，柱状图-横...）
    label → 图例标签，Dataframe格式以列名为label
    style → 风格字符串，这里包括了linestyle（-），marker（.），color（g）
    color → 颜色，有color指定时候，以color颜色为准
    alpha → 透明度，0-1
    use_index → 将索引用为刻度标签，默认为True
    rot → 旋转刻度标签，0-360
    grid → 显示网格，一般直接用plt.grid
    xlim,ylim → x,y轴界限
    xticks,yticks → x,y轴刻度值
    figsize → 图像大小
    title → 图名
    legend → 是否显示图例，一般直接用plt.legend()
"""


def fun_basic_draw():
    """
    pandas 数据绘图
    """
    ts = pd.Series(np.random.randn(1000),
                   index=pd.date_range('1/1/2000', periods=1000))  # pandas 时间序列
    ts = ts.cumsum()
    ts.plot(kind="line",
            label="what",
            style="--",
            color='g',
            alpha=0.4,
            use_index=True,
            rot=45,
            grid=True,
            ylim=[-50, 50],
            yticks=list(range(-50, 50, 10)),
            figsize=(8, 4),
            title='TEST_TEST',
            legend=True)
    plt.legend()

    # 四组数据画在一张图上
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                      columns=list('ABCD')).cumsum()
    df.plot(kind='line',
            style='--.',
            alpha=0.4,
            use_index=True,
            color="gbyr",
            rot=45,
            grid=True,
            figsize=(8, 4),
            title='test',
            legend=True,
            subplots=False,
            colormap='Greens')
    plt.show()


def draw_bar_pd_data():
    """
    柱状图
    """
    # 创建一个新的figure，并返回一个subplot对象的numpy数组
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    s = pd.Series(np.random.randint(0, 10, 16), index=list("abcdefghijklmnop"))
    df = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
    # print(df)

    # 单系列柱状图方法一：plt.plot(kind="bar/barh")
    s.plot(kind="bar", color="k", grid=True, alpha=0.5, ax=axes[0])  # ax参数 -> 选择第几个子图
    # plt.show()

    # 多系列柱状图
    # df.plot(kind="bar", ax=axes[1], grid=True, colormap="Reds_r")
    df.plot(kind="bar", ax=axes[1], grid=True, color="bgy")  # 颜色设置区别
    # print(df)
    # plt.show()

    # 多系列堆叠图: stacked -> 堆叠
    df.plot(kind="bar", ax=axes[2], grid=True, color="bgy", stacked=True)
    plt.show()


def draw_bar_plt():
    plt.figure(figsize=(10, 4))
    x = np.arange(10)
    y1 = np.random.rand(10)
    y2 = -np.random.rand(10)

    plt.bar(x, y1, width=1, facecolor="yellowgreen", edgecolor="white",
            yerr=y1 * 0.1)
    plt.bar(x, y2, width=1, facecolor='lightskyblue', edgecolor='white',
            yerr=y2 * 0.1)
    for i, j in zip(x, y1):
        plt.text(i - 0.2, j - 0.15, '%.2f' % j, color='white')
    for i, j in zip(x, y2):
        plt.text(i - 0.2, j + 0.05, '%.2f' % -j, color='white')
    plt.show()


"""
饼图
参数含义：
    第一个参数：数据
    explode：指定每部分的偏移量
    labels：标签
    colors：颜色
    autopct：饼图上的数据标签显示方式
    pctdistance：每个饼切片的中心和通过autopct生成的文本开始之间的比例
    labeldistance：被画饼标记的直径,默认值：1.1
    shadow：阴影
    startangle：开始角度
    radius：半径
    frame：图框
    counterclock：指定指针方向，顺时针或者逆时针
"""


def pie():
    """
    pie function
    """
    s = pd.Series(3 * np.random.rand(4), index=["a", "b", "c", "d"],
                  name="series")
    plt.axis("equal")  # 保证长宽相对
    plt.pie(
        s,
        explode=[0.1, 0, 0, 0],
        labels=s.index,
        colors="rgbc",
        autopct="%.2f%%",
        pctdistance=0.6,
        labeldistance=1.2,
        shadow=False,
        startangle=0,
        radius=1.5,
        frame=False
    )
    plt.show()
    # plt.savefig("tt.jpg")  # 保存这个地方和show有个坑


"""
散点图
参数含义：
    s：散点的大小
    c：散点的颜色
    vmin,vmax：亮度设置，标量
    cmap：colormap
"""


def point():
    """
    scatter function
    """
    plt.figure(figsize=(8, 6))

    x = np.random.randn(1000)
    y = np.random.randn(1000)

    plt.scatter(x, y, marker='.',
                s=np.random.randn(1000) * 100,
                cmap='Reds_r',  # 渐变色
                c=y,  # 渐变色
                alpha=0.8, )
    plt.grid()  # 画网格
    plt.show()


"""

"""


if __name__ == "__main__":
    # fun_basic_draw()
    # draw_bar_pd_data()
    # draw_bar_plt()
    # pie()
    point()