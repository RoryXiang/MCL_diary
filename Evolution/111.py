import numpy as np
a = np.array([[1, 6, 2], [4, 6, 8]])
# print(a)
b = np.array([8, 3, 4])
# print(2**np.arange(10)[::-1])
# print(float(2**10 - 1) * 8)
# print(b.shape, type(b), type(a))
# print(a.dot(b))
# print(a.shape, b.shape)
# print(np.random.choice(5, 3, replace=True))
# a = np.array([1, 2, 3, 4, 5])
# b = [True, False, False, False, True]
# c = np.array([99, 88])
# a[b] = c
# print(a)
a = [[1, 2, 3], [4, 5, 6]]
for m in a:
    m[1] = m[1] ** 2
print(a)
