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


# create_plot()


def get_numleafs(my_tree):
    leaf_nums= 0
    first_str = list(my_tree.keys())[0]
    second_dic = my_tree[first_str]
    for key in second_dic:
        if type(second_dic[key]).__name__ == "dict":
            leaf_nums += get_numleafs(second_dic[key])
        else:
            leaf_nums += 1
    return leaf_nums


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dic = my_tree[first_str]
    for key in second_dic:
        if type(second_dic[key]).__name__ == "dict":
            this_depth = 1 + get_numleafs(second_dic[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    list_of_tree = [{"no sufacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                    {"no sufacing": {0: "no", 1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    return list_of_tree[i]


my_tree = retrieve_tree(0)
print(my_tree)
leafs = get_numleafs(my_tree)

print(leafs)

depth = get_tree_depth(my_tree)

print(depth)