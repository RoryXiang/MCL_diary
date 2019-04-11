import numpy as np

a = [1, 3]
b = np.array([-7, 4, 5, 77, 9, 22, 41, 32])
print(np.clip(b, *a))
print(b.argsort()[-a:])
good_idx = idx[fitness.argsort()][-POP_SIZE:]
