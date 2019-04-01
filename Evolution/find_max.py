import numpy as np
import matplotlib.pyplot as plt


DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.7         # mating probability (DNA crossover)
MUTATION_RATE = 0.005    # mutation probability
N_GENERATIONS = 300
X_BOUND = [0, 8]         # x upper and lower bounds


def f(x):
    """计算单个个体的适应性（y值）"""
    return np.cos(2 * x) * x + np.sin(10 * x) * x


def translate_DNA(pop):
    """
    将二进制转换成十进制：下面函数的意义是：两个矩阵相乘，pop矩阵乘以转换矩阵--->pop.dot(trans_array) --> return 公式讲解： [1,0,1,1,0] * [[2**4], [2**3], [2**2], [2**1], [2**0]] (/前半部分的公式得到10进制数)，然后除以2**5（2**DNA_SIZE）将10进制数转换成小于1的数，然后乘以8转换成8以内的数
    """
    # print(float(2**DNA_SIZE - 1) * X_BOUND[1])
    # print(pop.dot(2 ** np.arange(DNA_SIZE)[::-1]))
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE - 1) * X_BOUND[1]
    pass


def get_fitness(pred):
    """
    获取每个对象的适应性，这里直接返回y值，但是会有负数，但是在下一代选择的时候需要用到概率，而概率不能为负数，所以这里需要正数处理：减去数组中最小数然后加上0.001
    """
    return pred + 1e-3 - np.min(pred)


def select(pop, fitness):
    """
    # 选择函数:
        选择适应性高的个体留下来
    """
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    print(idx)
    return pop[idx]


def crossover(parent, pop):
    """
    # 交叉配对函数
    交配的并不是前多少用父亲后多少用母亲，二十随机index用，实现方法就如下demo：
        a = np.array([1, 2, 3, 4, 5])
        b = [True, False, False, False, True]
        c = np.array([99, 88])
        a[b] = c
        print(a)
    """
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(
            np.bool)   # choose crossover points
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child):
    """
    # 变异函数 随机一位上产生变异
    """
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


def main():
    # 2 表示2进制，size表示响向量的维（m*n)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))
    # print(pop.dot(2 ** np.arange(DNA_SIZE)
    #               [::-1]) / float(2**DNA_SIZE - 1) * X_BOUND[1])
    # print(pop)  # pop 其实就是x坐标，这里只是将数据转换成2进制
    plt.ion()
    # x = np.linspace(*X_BOUND, 200)
    x = np.linspace(start=X_BOUND[0], stop=X_BOUND[1], num=200)  # 生成x变量
    # print(x)
    plt.plot(x, f(x))
    plt.show()
    for _ in range(N_GENERATIONS):
        values = f(translate_DNA(pop))
        # if 'sca' in globals():
        try:
            sca.remove()
        except Exception as e:
            pass
        sca = plt.scatter(translate_DNA(pop), values,
                          s=200, lw=0, c='red', alpha=0.5)
        plt.pause(0.05)
        fitness = get_fitness(values)
        print("Most fitted DNA: ", pop[np.argmax(fitness), :])
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child       # parent is replaced by its child
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
    # # fig = plt.ion()
    # x = np.linspace(*X_BOUND, 200)
    # plt.plot(x, f(x))
    # plt.show()
