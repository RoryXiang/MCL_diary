# coding=utf-8
import numpy as np
import math

a = np.array([1, 3, 4])
b = np.array([[2, 4, 7], [6, 7, 8], [8, 2, 6]])

delta = b - a
print(np.argmin(delta, axis=0))
print(delta)
# print(delta @ delta)
# print(delta * delta)
# print(np.sum(delta @ delta, axis=1))
# print(np.sum(delta * delta, axis=1))
