import matplotlib.pylab as plt


# boxstyle文本框样式， fc(face color)背景透明度
# decision_node = dict(boxstyle="round4, pad=0.5", fc="0.8")
decision_node = dict(boxstyle="sawtooth", fc="0.8")
# leaf_node = dict(boxstyle="circle", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
# 箭头样式
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.axl.annotate(node_txt, xy=parent_pt, xycoords="axes fraction",
                             xytext=center_pt, textcoords="axes fraction",
                             va="center", ha="center", bbox=node_type,
                             arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    create_plot.axl = plt.subplot(111, frameon=False)
    plot_node("决策节点", (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node("叶节点", (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


create_plot()
