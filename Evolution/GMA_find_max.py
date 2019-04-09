"""
微生物遗传算法。其核心是只在比较中失败者做基因交叉，将胜利者的一部分基因遗传给失败者，也只在失败者中产生基因突变
"""

import numpy as np
import matplotlib.pyplot as plt


DNA_SIZE = 10           # DNA 长度
POP_SIZE = 20           # 种群数量
CROSS_RATE = 0.6        # 交叉配对概率
MUTATION_RATE = 0.01    # 产生变异的概率
N_GENERATIONS = 300
X_BOUND = [0, 5]        # x元素的上下界


def f(x):
    """计算单个个体的适应性（y值）"""
    return np.cos(2 * x) * x + np.sin(10 * x) * x


class MGA(object):
    """docstring for MGA"""

    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, population_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.population_size = population_size

        # 初始化种群
        self.population = np.random.randint(
            *DNA_bound, size=(self.population_size, self.DNA_size))

    def translate_DNA(self, population):
        """
        将二进制转换成十进制：下面函数的意义是：两个矩阵相乘，pop矩阵乘以转换矩阵--->pop.dot(trans_array) --> return 公式讲解： [1,0,1,1,0] * [[2**4], [2**3], [2**2], [2**1], [2**0]] (/前半部分的公式得到10进制数)，然后除以2**5（2**DNA_SIZE）将10进制数转换成小于1的数，然后乘以8转换成8以内的数
        """
        # print(float(2**DNA_SIZE - 1) * X_BOUND[1])
        # print(pop.dot(2 ** np.arange(DNA_SIZE)[::-1]))
        return population.dot(2 ** np.arange(self.DNA_size)[::-1]) / float(2 ** self.DNA_size - 1) * X_BOUND[1]

    def get_fitness(self, value):
        return value

    def crossover(self, loser_winner):
        """交叉函数。在随机的DNA位置上将loser的DNA换成winner的
        Arguments:
            loser_winner {[type]} -- [description]
        Returns:
            [type] -- [description]
        """
        cross_point = np.empty((self.DNA_size,), dtype=np.bool)
        for i in range(self.DNA_size):
            cross_point[i] = True if np.random.randn(
            ) < self.cross_rate else False
        loser_winner[0, cross_point] = loser_winner[1, cross_point]
        return loser_winner

    def mutate(self, loser_winner):
        """变异函数，只在loser的基因上产生变异
        Arguments:
            loser_winner {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        mutation_point = np.empty((self.DNA_size,), dtype=np.bool)
        for i in range(self.DNA_size):
            mutation_point[i] = True if np.random.randn(
            ) < self.mutate_rate else False
        loser_winner[0, mutation_point] = ~loser_winner[0,
                                                        mutation_point].astype(np.bool)
        return loser_winner

    def evolve(self, n):
        for _ in range(n):  # random pick and compare n times
            sub_pop_idx = np.random.choice(
                np.arange(0, self.population_size), size=2, replace=False)
            sub_pop = self.population[sub_pop_idx]             # pick 2 from pop
            product = f(self.translate_DNA(sub_pop))
            fitness = self.get_fitness(product)
            loser_winner_idx = np.argsort(fitness)
            # the first is loser and second is winner
            loser_winner = sub_pop[loser_winner_idx]
            loser_winner = self.crossover(loser_winner)
            loser_winner = self.mutate(loser_winner)
            self.population[sub_pop_idx] = loser_winner
        DNA_prod = self.translate_DNA(self.population)
        pred = f(DNA_prod)
        return DNA_prod, pred


if __name__ == '__main__':
    plt.ion()       # something about plotting
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, f(x))

    ga = MGA(DNA_size=DNA_SIZE, DNA_bound=[
             0, 1], cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE, population_size=POP_SIZE)

    for _ in range(N_GENERATIONS):                    # 100 generations
        # natural selection, crossover and mutation
        DNA_prod, pred = ga.evolve(5)

        # something about plotting
        if 'sca' in globals():
            sca.remove()
        sca = plt.scatter(DNA_prod, pred, s=200, lw=0, c='red', alpha=0.5)
        plt.pause(0.05)

    plt.ioff()
    plt.show()
