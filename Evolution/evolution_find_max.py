import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1
DNA_BOUND = [0, 5]
N_GENERATION = 200
POPULATION_SIZE = 100
KID_NUMBER = 50


def f(x):
    # to find the maximum of this function
    return np.sin(10 * x) * x + np.cos(2 * x) * x


def get_fitness(pred):
    return pred.flatten()


def make_chid(population, KID_NUMBER):
    kids = {"DNA": np.empty((KID_NUMBER, DNA_SIZE))}
    kids["mut_strength"] = np.empty_like(kids["DNA"])
    for kid_value, kid_mutation in zip(kids["DNA"], kids["mut_strength"]):
        # 选择出父母(下标)
        father, mother = np.random.choice(
            np.arange(POPULATION_SIZE), size=2, replace=False)
        # 选择交叉点
        cross_point = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)
        # 将父母的DNA遗传给孩子（包括每个DNA的变异率）
        kid_value[cross_point] = population["DNA"][father, cross_point]
        kid_value[~cross_point] = population["DNA"][mother, ~cross_point]
        kid_mutation[cross_point] = population["mut_strength"][father, cross_point]
        kid_mutation[~cross_point] = population["mut_strength"][mother, ~cross_point]

        # mutate 一个个体
        # 先将变异系数产生变异：将原来的变异系数加上一个-0.5~0.5的随机值，如果小于0，则取0（np.maximum实现）？？？如何实现变异系数收敛的？？？
        kid_mutation = np.maximum(
            kid_mutation + (np.random.rand(*kid_mutation.shape) - 0.5), 0.)
        # 将DNA产生变异: 在原来的孩子DNA基础上加上变异率乘以一个正太分布的系数
        kid_value += kid_mutation * np.random.randn(*kid_value.shape)
        # 上一步DNA产生变异后，DNA有可能跑出DNA限制的范围，需要通过下一步来限制
        kid_value[:] = np.clip(kid_value, *DNA_BOUND)
        # kid_value = np.clip(kid_value, *DNA_BOUND)
        print(kid_value)
    print(kids["DNA"])
    return kids


def kill_bad(population, kids):
    # 将新生成的孩子放到种群中
    # print(population)
    # print(kids)
    for key in ["DNA", "mut_strength"]:
        population[key] = np.vstack((population[key], kids[key]))
    # 计算所有个体的fitness
    fitness = get_fitness(f(population["DNA"]))
    index = np.arange(population["DNA"].shape[0])
    # 筛选出排序靠后的个体（排序靠后，fitness越高）
    good_index = index[fitness.argsort()][-POPULATION_SIZE:]
    for key in ["DNA", "mut_strength"]:
        population[key] = population[key][good_index]
    return population


# 初始化种群
population = dict(DNA=5 * np.random.rand(POPULATION_SIZE, DNA_SIZE),
                  mut_strength=np.random.rand(POPULATION_SIZE, DNA_SIZE))

plt.ion()
x = np.linspace(*DNA_BOUND, 200)
plt.plot(x, f(x))

for _ in range(N_GENERATION):
    try:
        sca.remove()
    except Exception as e:
        pass
    sca = plt.scatter(population["DNA"], f(
        population["DNA"]), lw=0, c='red', alpha=0.5)
    plt.pause(0.05)
    kids = make_chid(population, KID_NUMBER)
    population = kill_bad(population, kids)
