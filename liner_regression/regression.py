import numpy as np
import matplotlib.pyplot as plt


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


# 标准线性回归
def stand_regres(xarr, yarr):
    """
    :param xarr:  特征值
    :param yarr: 目标值
    :return:
    """
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    xTx = x_mat.T*x_mat
    if np.linalg.det(xTx) == 0:
        print("This is singular, cant not do inverse")
        return
    ws = np.linalg.solve(xTx, x_mat.T*y_mat)
    # ws = xTx.I * x_mat.T * y_mat
    return ws


xarr, yarr = load_data("ex0.txt")
# print(xarr[0:2])

ws = stand_regres(xarr, yarr)
# print(ws)

x_mat = np.mat(xarr)
y_mat = np.mat(yarr)
y_hat = x_mat*ws

# 把数据和预测直线画出来-------------------------------------------------------
#
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_mat[:,1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])  # numpy 的操作
x_copy = x_mat.copy()
x_copy.sort(0)
y_hat = x_copy*ws
ax.plot(x_copy[:,1], y_hat)
plt.show()

# -----------------------------------------------------------------------------


# 局部加权线性回归  k值
# 增加了核函数，使用高斯核函数，相当于只用于当前数据点相近的部分数据计算回归系数
def lwlr(test_point, xarr, yarr, k=1.0):
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    m = np.shape(x_mat)[0]
    weigths = np.mat(np.eye((m)))  # 创建对角矩阵
    for j in range(m):
        diff_mat = test_point - x_mat[j]
        weigths[j, j] = np.exp(diff_mat*diff_mat.T/(-2.0*k**2))  # 权重值大小以指级衰减
    xTx = x_mat.T * (weigths*x_mat)
    if np.linalg.det(xTx) == 0:
        print("This is singular, cant not do inverse")
        return
    ws = xTx.I * x_mat.T * weigths * y_mat  # TODO ????
    return test_point * ws

print(yarr[0])


def lwlr_test(testarr, xarr, yarr, k=1.0):
    m = np.shape(testarr)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(testarr[i], xarr, yarr, k)
    return y_hat


#
y_hat = lwlr_test(xarr, xarr, yarr, 0.01)
x_mat = np.mat(xarr)
srt_ind = x_mat[:, 1].argsort(0)
x_sort = x_mat[srt_ind][:, 0, :]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_sort[:, 1], y_hat[srt_ind])
ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(yarr).T.flatten().A[0], s=2, c="red")
plt.show()

print(np.mat(np.array([1,2,3]))*np.mat(np.array([1,2,3])).T)
print(np.sum(np.mat(np.array([1,2,3]))*np.mat(np.array([1,2,3])).T))
