import numpy as np
import matplotlib.pyplot as plt
# a = np.array([[1, 6, 2], [4, 6, 8]])
# print(a)
# b = np.array([8, 3, 4])
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
# a = [[1, 2, 3], [4, 5, 6]]
# for m in a:
#     m[1] = m[1] ** 2
# print(a)

# a = np.array([1, 2, 3, 4, 5, 6])
# b = np.array([True, False, False, False, False, True])
# c = [1, 0, 0, 0, 0, 1]
# print((a == b).sum(axis=0))
# print(a[b])
# print(a[c])
# print(np.random.randint(
#     0, 2, 10, np.bool))


# a = np.array([1, 2, 3, 4, 5, 6])
# b = [2, 2, 2, 2, 2, 2]
# b[:] = a
# print(b)
# d = [1, 2, 3, 4, 5]
# a = np.array([d for _ in range(10)])
# b = np.vstack([d for _ in range(10)])
# print(a.shape, b.shape)
# print((a == b).all())

# a = [1, 2, 4, 6]
# b = np.empty_like(a, dtype=np.float64)
# print(b)
# c = np.empty(b.shape, dtype=np.float64)
# print(c)
xs = [0, 3, 6, 9]
ys = [0, 4, 8, 12]
xs = [0, 2, 4, 8]

# mm = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
# print(np.diff(xs, n=2))
# print(np.square(np.diff(xs)))
# print(np.square(np.diff(xs)) + np.square(np.diff(ys)))

# print(np.exp(2 * 2 / a))
# print(2 * 2 / a)
#
# x = np.linspace(-5, 5)


# def f(x):
#     return np.exp(x)


# plt.plot(x, f(x))
# plt.show()

a = np.array([1, 2, 3, 4, 5, 6])
print(a.ravel())
c = np.array([6, 3, 1, 4, 2, 5])
b = np.array([True, False, True, False, False, True])
mm = a[b]
nn = c[~np.isin(c, mm)]
print(mm)
print(np.isin(c, mm))
print(~np.isin(c, mm))
print(nn)
print(np.concatenate((mm, nn)))
# print(*[2, 3])


# print(np.random.randn()
#       )
