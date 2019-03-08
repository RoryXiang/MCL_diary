import math
import numpy as np


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
    return 1.0 / (1 + np.exp(-in_x))


def grad_ascent(data_mat, class_labels):
    pass
