from math import log
import math


def clacShannonEnt(data):
    """
    计算信息熵/calculate information entropy
    :param data:
    :return:
    """
    numentries = len(data)
    labelcounts = {}
    for feattvec in data:
        currentlabel = feattvec[-1]
        if currentlabel not in labelcounts:
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
    shannonent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key])/numentries
        shannonent -= prob*log(prob, math.e)
    return shannonent


def create_data():
    data = [
        [1, 1, "yes"],
        [1, 1, "yes"],
        [1, 0, "no"],
        [0, 1, "no"],
        [0, 1, "no"],
    ]
    labels = ["no surfacing", "flippers"]
    return data, labels


data, labels = create_data()

score = clacShannonEnt(data)
print(score)

data[0][-1] = "mabey"

score = clacShannonEnt(data)
print(score)


def split_data(data, axis, value):
    """
    划分数据集，即返回符合value的所有数据特征（去除掉axis对应的特征值）
    :param data: 带划分数据集
    :param axis: 特征
    :param value: 需要返回的特征值
    :return:
    """
    tar_data = []
    for featvec in data:
        if featvec[axis] == value:
            reduce_featvec = featvec[:axis]
            reduce_featvec.extend(featvec[axis+1:])
            tar_data.append(reduce_featvec)
    return tar_data


def choose_bast_feature_to_split(data):
    """
    找到理想特征。思想是，针对每一特征计算他们的特征信息熵，
    取信息熵最大者作为最优特征
    ?????问题，信息熵越大代表什么，越小代表什么？？？？？？
    :param data:
    :return:
    """
    feature_num = len(data[0])
    base_entropy = clacShannonEnt(data)  # 计算信息熵
    beast_info_gain = 0.0
    beast_feature = -1
    for i in range(feature_num-1):
        # 创建唯一的分类标签列表-------------------------
        feature_list = [example[i] for example in data]
        unique_values = set(feature_list)
        new_entropy = 0.0
        # 计算第i个特征每个特征值的信息熵----------------
        for value in unique_values:
            sub_data = split_data(data, i, value)
            prob = len(sub_data)/float(len(data))
            new_entropy += prob*clacShannonEnt(sub_data)  # TODO
        info_gain = base_entropy - new_entropy
        # 如果此信息的信息熵比之前的高，则代替之前的-----
        if info_gain > beast_info_gain:
            beast_info_gain = info_gain
            beast_feature = i
    return beast_feature


data, _ = create_data()
feature = choose_bast_feature_to_split(data)
print(feature)


import operator


def majority_cnt(class_list):
    # TODO ???
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = \
        sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data, labels):
    """
    构造决策树。
    :param data:
    :param labels:
    :return:
    """
    class_list = [example[-1] for example in data]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data[0]) == 1:
        return majority_cnt(class_list)
    beast_feat = choose_bast_feature_to_split(data)
    baset_feat_label = labels[beast_feat]
    my_tree = {baset_feat_label: {}}
    del(labels[beast_feat])
    feat_values = [example[beast_feat] for example in data]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[baset_feat_label][value] = \
            create_tree(split_data(data, beast_feat, value), sub_labels)
    return my_tree


data, labels = create_data()

my_tree = create_tree(data, labels)

print(my_tree)