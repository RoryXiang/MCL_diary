import numpy as np
from liner_regression.regression import load_data, stand_regres
import matplotlib.pyplot as plt


# 岭回归----------------------------------------------------
def rigre_regression(x_mat, y_mat, lam=0.2):
    xTx = x_mat.T*x_mat
    denom = xTx + np.eye(np.shape(x_mat)[1]) * lam
    if np.linalg.det(denom) == 0:
        print("This is singular, cant not do inverse")
        return
    ws = denom.I * x_mat.T * y_mat
    return ws


def ridge_test(xarr,yarr):
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean     # to eliminate X0 take mean off of Y
    x_means = np.mean(x_mat, 0)   # calc mean then subtract it off
    x_var = np.var(x_mat, 0)      # calc variance of Xi then divide by it
    x_mat = (x_mat - x_means)/x_var
    numTestPts = 30
    w_mat = np.zeros((numTestPts, np.shape(x_mat)[1]))
    for i in range(numTestPts):
        ws = rigre_regression(x_mat, y_mat, np.exp(i-10))
        w_mat[i] = ws.T
    return w_mat


def regularize(x_mat):#regularize by columns
    in_mat = x_mat.copy()
    in_means = np.mean(in_mat, 0)   #calc mean then subtract it off
    in_var = np.var(in_mat, 0)      #calc variance of Xi then divide by it
    in_mat = (in_mat - in_means)/in_var
    return in_mat


def plot():
    abx, aby = load_data("./abalone.txt")
    w_ = ridge_test(abx, aby)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 展示结果-------------------------------------
    # x_mat = np.mat(abx)
    # y_hat = w_ * x_mat
    # ax.scatter(x_mat[:, 1].flatten().A[0], y_hat.T[:, 0].flatten().A[0])
    # x_copy = x_mat.copy()
    # x_copy.sort(0)
    # ax.plot(x_copy[:, 1], y_hat)
    # ----------------------------------------------

    ax.plot(w_)
    plt.show()


if __name__ == '__main__':
    plot()


# 计算平方误差
def rss_error(yarr, yhat):
    return ((yarr-yhat)**2).sum()


# 前向逐步回归--------------------------------------------------------------
def stage_wise(xarr, yarr, eps=0.01, num_it=100):
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = (x_mat - np.mean(x_mat, 0)) / np.var(x_mat, 0)
    # x_mat = regularize(x_mat)
    m, n = np.shape(x_mat)
    return_mat = np.zeros((num_it, n))
    ws = np.zeros((n, 1))
    ws_test = ws_max = ws.copy()
    for i in range(num_it):
        lowest_error = np.inf
        for j in range(n):
            # 每个特征值的权重，尝试增加或减小一定的数值来查看平方误差
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps*sign
                y_test = x_mat *ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:
                    ws_max = ws_test
        ws = ws_max.copy()
    return return_mat


def show__():
    xarr, yarr = load_data("./abalone.txt")
    a = stage_wise(xarr, yarr, 0.01, 5000)
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    x_mat = regularize(x_mat)
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    weights = stand_regres(x_mat, y_mat.T)
    print(weights.T)


if __name__ == '__main__':
    show__()