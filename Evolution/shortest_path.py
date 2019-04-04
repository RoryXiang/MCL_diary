import numpy as np
import matplotlib.pyplot as plt

CITY_NUM = 20
CROSS_RATE = 0.3
MUTATION_RATE = 0.01
POP_SIZE = 500
N_GENERATION = 1000


class GA(object):
    """docstring for GA"""

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
        """将所有DNA翻译成坐标，[[1,6,9,10]] --> x: [[2.3, 4.6, 5.1, 9.0]]; y: [[1.3, 6.6, 1.1, 2.2]]
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
            x_values[index] = city_position[d_number][:, 0]
            y_values[index] = city_position[d_number][:, 1]
        return x_values, y_values

    # ==============================================

    def get_fitness(self, xs_values, ys_values):
        """计算每条路线的距离distance，并且以距离的倒数为e的指数的值作为fitness
        Arguments:
            xs_values [[float]]-- 所有路线的x坐标列表
            ys_values [[float]]-- 所有路线的y坐标列表

        Returns:
            fitness [float] -- 每条路线的适应性
            total_distance [float] -- 每条路线的距离
        """
        total_distance = np.empty((xs_values.shape[0], ), dtype=np.float64)
        for index, (x_values, y_values) in enumerate(zip(xs_values, ys_values)):
            total_distance[index] = np.sum(
                np.sqrt(np.square(np.diff(x_values)) + np.square(np.diff(y_values))))
        # fitness = np.exp(self.DNA_size * 2 / total_distance)
        fitness = np.exp(1 / total_distance)
        return fitness, total_distance

    def select(self, fitness):
        """选择函数
        """
        print(fitness.shape)
        # print("???????", fitness)
        index = np.random.choice(np.arange(
            self.people_size), size=self.people_size, replace=True, p=fitness / fitness.sum())
        return self.people[index]

    def crossover(self, father, people):
        import time
        """交叉配对函数。原理是，选择父亲的一部分城市排序，然后剩余的城市排序按照母亲的排序方式组合成孩子路线方案
        Arguments:
            father [city1,city2..] -- 所有的城市排序
            people [[city1,city2..],[city4,city10..]] -- 所有的路线方案

        Returns:
            [city1,city2..] -- 生成的孩子路线方案或者父亲路线
        """
        if np.random.rand() < self.cross_rate:
            # 从people中选择另一个parent
            mother_index = np.random.randint(0, self.people_size, 1)
            # 选择交叉的DNA下标
            cross_points = np.random.randint(0, 2, self.DNA_size, dtype=np.bool)
            # 选择father部分的城市（排序方式为father已有的排序）
            father_city = father[cross_points]
            # 选择father没选上的城市（排序方式为mother以后的排序）
            mother_city = people[mother_index, ~np.isin(
                people[mother_index].ravel(), father_city)]
            
            child = np.concatenate((father_city, mother_city))
            # time.sleep(10)
            father[:] = np.concatenate((father_city, mother_city))
            return father
            return child
        return father

    def mutate(self, child):
        """变异函数。每个城市都有几率产生变异。变异：随机与路线上的另外一个城市交换位置
        Arguments:
            child [int] -- 路线方案

        Returns:
            child [int] -- 路线方案
        """
        for point_index in range(self.DNA_size):
            if np.random.rand() < self.mutation_rate:
                swap_point_index = np.random.randint(0, self.DNA_size)
                swap_A, swap_B = child[point_index], child[swap_point_index]
                child[point_index], child[swap_point_index] = swap_B, swap_A
        return child

    def evolve(self, fitness):
        """进化函数（一次进化）
        Arguments:
            fitness {[type]} -- [description]
        """
        people = self.select(fitness)
        people_copy = people.copy()
        for father in people:
            child = self.crossover(father, people_copy)
            child = self.mutate(child)
            father = child
        self.people = people


class TSP(object):
    """docstring for TSP"""

    def __init__(self, city_num):
        self.city_position = np.random.rand(city_num, 2)
        # print("?????", self.city_position)
        plt.ion()

    def ploting(self, x_values, y_values, total_distance):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T,
                    self.city_position[:, 1].T, s=100, c="k")
        plt.plot(x_values.T, y_values.T, "r-")
        plt.text(-0.05, -0.05, "Total distance=%.2f" %
                 total_distance, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)


ga = GA(DNA_size=CITY_NUM, cross_rate=CROSS_RATE,
        mutation_rate=MUTATION_RATE, people_size=POP_SIZE)

env = TSP(city_num=CITY_NUM)

for generation in range(N_GENERATION):
    x_values, y_values = ga.translate_DNA(ga.people, env.city_position)
    fitness, total_distance = ga.get_fitness(x_values, y_values)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
    env.ploting(x_values[best_idx], y_values[best_idx],
                total_distance[best_idx])

plt.ioff()
plt.show()
