# coding=utf-8
"""
普通的k均值聚类
"""

import numpy as np
import matplotlib.pyplot as plt
import copy


def show_area(conteners, i, k):
    """
    将所有点按照聚类画出来
    """
    colors = ['black', 'red', 'blue', 'green', "purple", "yellow", "pink"]
    for m in range(k):
        xs = [pp[0] for pp in conteners[m]]
        ys = [pp[1] for pp in conteners[m]]
        plt.scatter(xs, ys, c=colors[m])
    plt.title(f'iter {i}', fontsize=20)
    plt.pause(0.5)


def new_center(conteners, k):
    """
    计算新的中心点
    """
    centers = [[0, 0] for _ in range(k)]

    for k_index, p_set in enumerate(conteners):
        p_xs = [p[0] for p in p_set]
        p_ys = [p[1] for p in p_set]
        if len(p_set) == 0:
            # 当有一个类里面没有点的时候
            a = np.random.choice(len(points), size=1)
            new = copy.deepcopy(points[a])
            new[0] = new[0] + 0.1
            new[1] = new[1] + 0.1
            centers[k_index] = new
            continue
        centers[k_index][0] = sum(p_xs) / len(p_set)
        centers[k_index][1] = sum(p_ys) / len(p_set)
    return np.array(centers)


if __name__ == '__main__':
    points = np.random.uniform(0, 20, (50, 2))
    x = [points[i][0] for i in range(50)]
    y = [points[i][1] for i in range(50)]
    plt.scatter(x, y)
    plt.title('all points', fontsize=20)
    plt.pause(2)
    k = 7
    # 随机抽取4个点作为初始中心
    centers_index = np.random.choice(len(points), size=k)
    centers = points[centers_index]
    center_x = [centers[i][0] for i in range(len(centers))]
    center_y = [centers[i][1] for i in range(len(centers))]
    plt.scatter(center_x, center_y, c='red')
    plt.pause(1)
    for i in range(10):
        conteners = [[] for _ in range(k)]
        for point in points:
            nearest_index = np.argmin(
                np.sum((centers - point) ** 2, axis=1) ** 0.5)
            conteners[nearest_index].append(point)

        plt.clf()  # 清空画布
        show_area(conteners, i, k)
        centers = new_center(conteners, k)
    # plt.show()

    from sklearn.cluster import KMeans

    loss = []

    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, max_iter=100).fit(points)
        loss.append(kmeans.inertia_ / len(points) / k)
    plt.clf()  # 清空画布
    plt.plot(range(1, 10), loss)
    plt.xlabel("k")
    plt.ylabel("loss")
    plt.title("k loss relation!")
    plt.show()
