"""蚂蚁算法解决TSP问题
    核心需要理解蚂蚁在行经中下一个point选择概率函数；以及影响概率函数的信息素的更新
"""

import numpy as np
import sys
import random
import math
import tkinter
import threading
from functools import reduce
import copy

ALPHA = 1.0  # 信息启发因子
BATA = 1.0  # 期望值启发因子
RHO = 0.5  # 信息素挥发因素
Q = 1  # 信息素总量
city_num = 50  # 城市总数
ant_num = 50  # 蚁群数量

# 初始化城市位置信息
city_positions = np.random.randint(0, 700, size=(city_num, 2))
# 初始化距离图谱
distance_graph = np.zeros((city_num, city_num))
# 初始化信息素
pheromone_graph = np.ones((city_num, city_num))


# 蚂蚁类=========================================


class Ant(object):
    """docstring for ClassName"""

    def __init__(self, ID):
        self.ID = ID
        self.__init_data()  # 初始化蚂蚁，选择出生点

    def __init_data(self):
        self.path = []  # 蚂蚁的路经
        self.callable_city = np.ones((city_num,), dtype=np.bool)  # 城市可访问状态
        self.total_distance = 0.0  # 蚂蚁的路径总距离
        self.current_city = random.randint(0, city_num - 1)  # 选择初始化城市
        self.path.append(self.current_city)
        self.callable_city[self.callable_city] = False
        self.move_count = 1

    def __choice_next_city(self):
        # 初始化转移概率矩阵（从当前城市到每一个城市的概率）
        cities_prob = np.zeros(shape=(city_num,))
        total_prob = 0.0
        # 用概率函数计算概率矩阵(联想选路概率函数)
        for next_city in range(city_num):
            if self.callable_city[next_city]:
                try:
                    cities_prob[next_city] = np.power(
                        pheromone_graph[self.current_city][next_city], ALPHA) * np.power(1.0 / distance_graph[self.current_city][next_city], BATA)
                    total_prob += cities_prob[next_city]
                except ZeroDivisionError as e:
                    print(
                        f"Ant ID: {self.ID}, current city: {self.current_city}, target city: {next_city}")
                    sys.exit(1)

        # 选择下一个城市 （貌似用轮盘算法比用numpy快）
        # next_city = np.random.choice(np.arange(city_num), size=1, p=cities_prob / float(total_prob))

        # 轮盘算法
        next_city = None
        if total_prob > 0:
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.callable_city[i]:
                    temp_prob -= cities_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        # 第一个循环的时候，只能随机选择
        if not next_city:
            next_city = random.randint(0, city_num - 1)
            while not self.callable_city[next_city]:
                next_city = random.randint(0, city_num - 1)
        return next_city

    def __total_distance(self):
        """计算路径距离总和"""
        temp_distance = 0.0
        for index in range(1, city_num):
            start, end = self.path[index], self.path[index - 1]
            temp_distance += distance_graph[start][end]

        # 回路 （这个地方根据不同的业务会有变动，比如旅游路线，是不需要回到原点的）
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    def __move(self, next_city):
        """移动到下一个城市"""
        self.path.append(next_city)
        self.callable_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    def __search_path(self):

        # 初始化蚂蚁信息
        self.__init_data()

        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)

        # 计算路径总长度(需要加上从最后一个城市回到第一个城市的距离)
        self.total_distance += distance_graph[self.path[-1]][self.path[0]]

        # self.__total_distance()  # 因为每一步move都做了distance add 所以没有必要浪费时间再重新计算前面的距离和，而是只要加上最后到第一个城市的距离就ok


# TSP ================================================
class TSP(object):
    """docstring for TSP"""

    def __init__(self, root, width=1000, height=800):

        # 创建tkinter画布
        self.root = root
        self.width = width
        self.height = height

        self.city_num = city_num
        self.canvas = tkinter.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="wite",
            xscrollincrement=1,
            yscrollincrement=1
        )
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.title("TSP蚁群算法(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 5
        self.__lock = threading.RLook()

        self.__bindEvents()
        self.new()

        for start in range(city_num):
            for end in range(city_num):
                temp_distance = pow((city_positions[end][0] - city_positions[start][0]), 2) + pow(
                    (city_positions[end][1] - city_positions[start][1]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[start][end] = float(int(temp_distance))

    def __bindEvents(self):
        """按键相应程序"""
        self.root.bind("q", self.quite)        # 退出程序
        self.root.bind("n", self.new)          # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)         # 停止搜索

    def title(self, s):
        """定义画布标题"""
        self.root.title(s)

    def new(self, evt=None):
        """初始化城市节点"""
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()  # 清除信息
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象

        # 初始化城市节点
        for index in range(city_num):
            x = city_positions[index][0]
            y = city_positions[index][1]
            self.nodes.append(
                (x, y))
            # 生产节点椭圆
            node = self.canvas.create_oval(
                x - self.__r,
                y - self.__r,
                x + self.__r,
                y + self.__r,
                fill="#ff0000",      # 填充红色
                outline="#000000",   # 轮廓白色
                tags="node",
            )
            self.nodes2.append(node)
            self.canvas.create_text(
                x, y - 10,
                text='(' + str(x) + ',' + str(y) + ')',
                fill="black"
            )
            # 初始化信息素
            for i in range(city_num):
                for j in range(city_num):
                    pheromone_graph[i][j] = 1.0

            self.ants = [Ant(ID) for ID in range(ant_num)]
            self.best_ant = self.ants[-1]
            self.best_ant.total_distance = 1 << 31  # 二进制左移
            self.iter = 1

        def line(self, order):
            # 删除原线
            self.canvas.delete("line")

            def line2(i1, i2):
                point1, point2 = self.nodes[i1], self.nodes[i2]
                self.canvas.create_line(
                    point1, point2, fill="#000000", tages="line")
                return i2
            reduce(line2, order, order[-1])

        def clear(self):
            """清除画布"""
            for item in self.canvas.find_all():
                self.canvas.delete(item)

        def quite(self, evt):
            self.__lock.acquire()
            self.__running = False
            self.__lock.release()
            self.root.destroy()
            print(u"\n程序已退出...")
            sys.exit()

        # 停止搜索
        def stop(self, evt):
            self.__lock.acquire()
            self.__running = False
            self.__lock.release()

        def search_path(self, evt=None):

            # 多线程
            self.__lock.acquire()
            self.__running = True
            self.__lock.release()
            while self.__running:
                for ant in self.ants:
                    ant.search_path()
                    if ant.total_distance < self.best_ant.total_distance:
                        self.best_ant = copy.deepcopy(ant)

                # 更新信息素
                self.__uodate_pheromone_graph()
