import numpy as np


def load_data(filename):
    feat_num = len(open(filename, "r").readline().split("\t")) - 1  # -1 不取最后一个值
    data_mat = []
    label_mat = []
    fr = open(filename, "r")
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split("\t")
        for i in range(feat_num):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regres(xarr, yarr):
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    xTx = x_mat.T*x_mat
    if np.linalg.det(xTx) == 0:
        print("This is singular, cant not do inverse")
        return
    ws = np.linalg.solve(xTx, x_mat.T*y_mat)
    return ws


xarr, yarr = load_data("ex0.txt")
print(xarr[0:2])

ws = stand_regres(xarr, yarr)
print(ws)

x_mat = np.mat(xarr)
y_mat = np.mat(yarr)
y_hat = x_mat*ws

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_mat[:,1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])  # numpy 的操作
x_copy = x_mat.copy()
x_copy.sort(0)
y_hat = x_copy*ws
ax.plot(x_copy[:,1], y_hat)
plt.show()