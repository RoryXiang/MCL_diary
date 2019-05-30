import tushare as ts
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

data = ts.get_hist_data('601518', start='2017-010-26',
                        end='2018-06-22', ktype='D')

datay = pd.DataFrame(
    columns=['滞后两天', '滞后一天', '当天涨跌情况'], index=range(len(data) - 2))
for i in range(2, len(data)):

    datay.iloc[i - 2, 0] = data.iloc[i - 2, 6]

    datay.iloc[i - 2, 1] = data.iloc[i - 1, 6]
    if data.iloc[i, 6] > 0:

        datay.iloc[i - 2, 2] = 1

    else:

        datay.iloc[i - 2, 2] = 0
print(datay)
x = np.array(np.array(datay.iloc[:, [0, 1]]).tolist())
bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=len(x))  # 设置均值漂移参数

m_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True)  # 计算聚类

m_estimator.fit(x)  # 训练均值漂移模型

labels = m_estimator.labels_  # 获取标记
cent = m_estimator.cluster_centers_  # 提取聚类的中心点位置
num_clumster = len(np.unique(labels))  # 计算聚群个数

print('聚群的个数为:', str(num_clumster))  # 显示
import matplotlib.pyplot as plt

from itertools import cycle

plt.figure()

markers = '.*xo+sp'  # 针对不同的群标记不一样的标记
colors = 'rgbkcmy'  # 针对不同的群标记不一样的颜色
# https://www.jianshu.com/p/b992c1279c73

for i, marker, color in zip(range(num_clumster), markers, colors):  # zip函数用于对应组合

    # 组合为（0,.）(1,*)(2,x)(3,v)
    print(i, marker, color)

    plt.scatter(x[labels == i, 0], x[labels == i, 1],
                marker=marker, color=color, s=30)  # 画出前7群点

    centr = cent[i]

    plt.plot(centr[0], centr[1], marker='o', markerfacecolor=color,
             markeredgecolor='k', markersize=8)

    plt.title('Clusters and their centroids')
    plt.pause(0.5)

plt.show()
