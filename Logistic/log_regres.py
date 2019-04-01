import math
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data_mat = []
    label_mate = []
    with open("./testSet.txt", "r", encoding="utf-8") as f:
        for line in f:
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mate.append(int(line_arr[2]))
    return data_mat, label_mate


def sigmoid(in_x):
    """Sigmoid函数"""
    return 1.0 / (1 + np.exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    """梯度上升算法
    Arguments:
        data_mat_in： {[type]} -- [description]
        class_labels {[type]} -- [description]
    Returns:
        [type] -- [description]
    """
    # 将数据转换成numpy矩阵
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001  # 向目标移动的步长
    max_cyless = 500  # 迭代次数
    weights = np.ones((n, 1))
    for k in range(max_cyless):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


if __name__ == '__main__':
    dataarr, labelmat = load_data()
    weights = grad_ascent(dataarr, labelmat)
    print(weights.getA())


def plot_best_fit(weights):
    datamat, label_mat = load_data()
    data_arr = np.array(datamat)
    n = np.shape(data_arr)[0]
    x_code1 = []
    y_code1 = []
    x_code2 = []
    y_code2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_code1.append(data_arr[i, 1])
            y_code1.append(data_arr[i, 2])
        else:
            x_code2.append(data_arr[i, 1])
            y_code2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_code1, y_code1, s=30, c="red", marker="s")
    ax.scatter(x_code2, y_code2, s=30, c="green")
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


if __name__ == '__main__':
    dataarr, labelmat = load_data()
    weights = grad_ascent(dataarr, labelmat)
    print(weights)
    plot_best_fit(weights.getA())
