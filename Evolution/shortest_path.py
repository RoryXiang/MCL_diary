import numpy as np
import matplotlib.pyplot as plt

CITY_NUM = 20
CROSS_RATE = 0.3
MUTATION_RATE = 0.01
POP_SIZE = 500
N_GENERATION = 1000


class TSP(object):
    """docstring for TSP"""

    def __init__(self, DNA_size, cross_rate, mutation_rate, people_size):
        self.DNA_size = DNA_size  # 城市数量
        self.cross_rate = cross_rate  # 交叉配对概率
        self.mutation_rate = mutation_rate  # 变异概率
        self.people_size = people_size  # 种群数量
        # 初始化第一代群体
        self.people = np.array([np.random.permutation(self.DNA_size)
                                for _ in range(self.people_size)])
        # self.people = np.vstack([np.random.permutation(self.DNA_size)
        #                          for _ in range(self.pop_size)])

    def translate_DNA(self, DNA, city_position):
        """将dna翻译成坐标，[1,6,9,10] --> x: [2.3, 4.6, 5.1, 9.0]; y: [1.3, 6.6, 1.1, 2.2]
        Arguments:
            DNA [int] -- 城市序号数组
            city_position [[float, float], [float, float]] -- 每个城市的坐标

        Returns:
            x_values [float] -- 所有城市的x坐标(一条路线的)
            y_values [float] -- 所有城市的y坐标
        """
        x_values = np.empty_like(DNA, dtype=np.float64)
        y_values = np.empty_like(DNA, dtype=np.float64)
        for index, d_number in enumerate(DNA):
            x_values[index] = city_position[d_number][0]
            y_values[index] = city_position[d_number][1]
        return x_values, y_values

    def get_fitness(self, xs_values, ys_values):
        total_distance = np.empty(xs_values.shape, dtype=np.float64)
        for index, (x_values, y_values) in enumerate(zip(xs_values, ys_values)):
            total_distance[index] = np.sum(
                np.sqrt(np.square(np.diff(x_values)) + np.square(np.diff(y_values))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance
