# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


def show_area(conteners, i):
    """
    将所有点按照聚类画出来
    """
    xs1 = [pp[0] for pp in conteners[0]]
    ys1 = [pp[1] for pp in conteners[0]]
    plt.scatter(xs1, ys1, c='black')
    xs2 = [pp[0] for pp in conteners[1]]
    ys2 = [pp[1] for pp in conteners[1]]
    plt.scatter(xs2, ys2, c='red')
    xs3 = [pp[0] for pp in conteners[2]]
    ys3 = [pp[1] for pp in conteners[2]]
    plt.scatter(xs3, ys3, c='blue')
    xs4 = [pp[0] for pp in conteners[3]]
    ys4 = [pp[1] for pp in conteners[3]]
    plt.scatter(xs4, ys4, c='green')
    plt.title(f'iter {i}', fontsize=20)
    plt.pause(3)


def new_center(conteners, k):
    """
    计算新的中心点
    """
    centers = [[0, 0] for _ in range(k)]

    for k_index, p_set in enumerate(conteners):
        p_xs = [p[0] for p in p_set]
        p_ys = [p[1] for p in p_set]
        centers[k_index][0] = sum(p_xs) / len(p_set)
        print("???", len(p_set))
        centers[k_index][1] = sum(p_ys) / len(p_set)
    return centers


if __name__ == '__main__':
    points = np.random.uniform(0, 20, (50, 2))
    x = [points[i][0] for i in range(50)]
    y = [points[i][1] for i in range(50)]
    plt.scatter(x, y)
    plt.title('all points', fontsize=20)
    plt.pause(2)
    k = 4
    # 随机抽取4个点作为初始中心
    centers_index = np.random.choice(len(points), size=k)
    centers = points[centers_index]
    center_x = [centers[i][0] for i in range(len(centers))]
    center_y = [centers[i][1] for i in range(len(centers))]
    print(centers)
    plt.scatter(center_x, center_y, c='red')
    plt.pause(1)
    for i in range(10):
        conteners = [[] for _ in range(k)]
        for point in points:
            nearest_index = np.argmin(
                np.sum((centers - point) ** 2, axis=1) ** 0.5)
            conteners[nearest_index].append(point)
        try:
            sca.remove()
        except Exception as e:
            pass
        show_area(conteners, i)
        centers = new_center(conteners, k)
    plt.show()
