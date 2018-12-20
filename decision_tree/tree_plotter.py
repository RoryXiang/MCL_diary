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


# my_tree = retrieve_tree(0)
# print(my_tree)
# leafs = get_numleafs(my_tree)
#
# print(leafs)
#
# depth = get_tree_depth(my_tree)
#
# print(depth)

# # 这个是用来绘制线上的标注，简单
def plot_mid_text(cntrpt, parentpt, txtsting):
    x_mid = (parentpt[0] - cntrpt[0]) / 2.0 + cntrpt[0]
    y_mid = (parentpt[1] - cntrpt[1]) / 2.0 + cntrpt[1]
    create_plot.axl.text(x_mid, y_mid, txtsting)


# # 重点，递归，决定整个树图的绘制，难（自己认为）
def plot_tree(my_tree, parentpt, node_txt):
    leaf_nums = get_numleafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntrpt = (plot_tree.x0ff + (1.0 + float(leaf_nums)) / 2.0 / plot_tree.totalw,
              plot_tree.y0ff)
    plot_mid_text(cntrpt, parentpt, node_txt)
    plot_node(first_str, cntrpt, parentpt, decision_node)

    second_dict = my_tree[first_str]
    print(second_dict)
    plot_tree.y0ff = plot_tree.y0ff - 1.0/plot_tree.totald
    for key in list(second_dict.keys()):
        if type(second_dict[key]).__name__ == "dict":   # test to see if the nodes are dictonaires, if not they are leaf nodes
            plot_tree(second_dict[key], cntrpt, str(key))
        else:  # it's a leaf node print the leaf node
            plot_tree.x0ff = plot_tree.x0ff + 1.0/plot_tree.totalw
            plot_node(second_dict[key], (plot_tree.x0ff, plot_tree.y0ff), cntrpt, leaf_node)
            plot_mid_text((plot_tree.x0ff, plot_tree.y0ff), cntrpt, str(key))
    plot_tree.y0ff = plot_tree.y0ff + 1.0 / plot_tree.totald


def create_plot(intree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.axl = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalw = float(get_numleafs(intree))
    plot_tree.totald = float(get_tree_depth(intree))
    plot_tree.x0ff = -0.5/plot_tree.totalw
    plot_tree.y0ff = 1.0
    plot_tree(intree, (0.5, 1.0), "")
    plt.show()


my_tree = retrieve_tree(0)
my_tree["no sufacing"][3] = "maybe"
print(my_tree)
create_plot(my_tree)
