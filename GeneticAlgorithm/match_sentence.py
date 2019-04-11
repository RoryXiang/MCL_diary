import numpy as np


TARTET_SENTENCE = "I LOVE YOU !"
poeple_SIZE = 300
CROSS_RATE = 0.5
MUTATION_RATE = 0.01
N_GENERATION = 1000

DNA_SIZE = len(TARTET_SENTENCE)
print("????", DNA_SIZE)
# convert string to number 将字符串转换成ASCII数组
TARGET_ASCII = np.fromstring(TARTET_SENTENCE, dtype=np.uint8)
# print(TARGET_ASCII)
# a = np.array([73, 32, 56, 79, 86, 77, 32, 89, 79, 85,
#               32, 84, 98, 32, 68, 69, 65, 111, 72, 32, 33])
# print((TARGET_ASCII == a).sum(axis=0))
# print(TARGET_ASCII.tostring().decode("ASCII"))  # 将ASCII转换成字符串
# print(1e-3)

ASCII_BOUND = [32, 126]


class GA(object):

    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, poeple_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.poeple_size = poeple_size

        # int8 for convert to ASCII
        self.poeple = np.random.randint(
            *DNA_bound, size=(poeple_size, DNA_size)).astype(np.int8)

    def translate_DNA(self, DNA):
        """将ASCII转换成字符串 convert to readable string
        Arguments:
            DNA {[type]} -- [int list]
        Returns:
            [type] -- string
        """
        return DNA.tostring().decode("ASCII")

    def get_fitness(self):
        """计算与目标字符串的相似度， count how many character matches
        Returns:
            int -- 有多少个字符是相似的
        """
        # self.poeple = [[23, 45, 78, ...], [67, 54, 121, ...]]每一行是一个个体，
        # 所以要针对每一行来计算相似度，所以axis=1表示安行计算
        match_count = (self.poeple == TARGET_ASCII).sum(axis=1)
        return match_count

    def select(self):
        """选择出匹配度高的poeple，按照概率选择。
        Returns:
            [type] -- [description]
        """
        # add a small amount to avoid all zero fitness
        fitness = self.get_fitness() + 1e-4
        index = np.random.choice(np.arange(
            self.poeple_size), self.poeple_size, replace=True, p=fitness / fitness.sum())
        return self.poeple[index]

    def crossover(self, parent, poeple):
        """
        # 交叉配对函数
        交配的并不是前多少用父亲后多少用母亲，二十随机index用，实现方法就如下demo：
            a = np.array([1, 2, 3, 4, 5])
            b = [True, False, False, False, True]
            c = np.array([99, 88])
            a[b] = c
            print(a)
        """
        if np.random.rand() < self.mutation_rate:
            index = np.random.randint(0, self.DNA_size, 1)
            # cross_points = np.random.randint(
            #     0, 2, self.DNA_size).astype(np.bool)
            cross_points = np.random.randint(
                0, 2, self.DNA_size, dtype=np.bool)
            parent[cross_points] = poeple[index, cross_points]
        return parent

    def mutate(self, child):
        """
        # 变异函数 每一位上都有机会产生变异产生变异
        """
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutation_rate:
                child[point] = np.random.randint(*self.DNA_bound)
                # child[point] = np.random.randint(self.DNA_bound[0], self.DNA_bound[1])
        return child

    def evolve(self):
        """进化函数
        """
        poeple = self.select()
        poeple_copy = poeple.copy()
        for parent in poeple:
            child = self.crossover(parent, poeple_copy)
            child = self.mutate(child)
            parent = child
        self.poeple = poeple


if __name__ == '__main__':
    ga = GA(DNA_size=DNA_SIZE, DNA_bound=ASCII_BOUND, cross_rate=CROSS_RATE,
            mutation_rate=MUTATION_RATE, poeple_size=poeple_SIZE)

    for generation in range(N_GENERATION):
        fitness = ga.get_fitness()
        best_DNA = ga.poeple[np.argmax(fitness)]
        best_phrase = ga.translate_DNA(best_DNA)
        print('Gen', generation, ': ', best_phrase)
        if best_phrase == TARTET_SENTENCE:
            break
        ga.evolve()
