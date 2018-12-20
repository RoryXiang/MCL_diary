import numpy as np
from liner_regression.regression import load_data

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
        ws = np.ridgeRegres(x_mat, y_mat, np.exp(i-10))
        w_mat[i,:] = ws.T
    return w_mat


abx, aby = load_data("./abalone.txt")
w_ = ridge_test(abx, aby)