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
    找到理想特征。思想是，针对每一特征计算他们的特征信息熵，取信息熵最大者作为最优特征
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
        # print("&&&&&", info_gain)
        # 如果此信息的信息熵比之前的高，则代替之前的-----
        if info_gain > beast_info_gain:
            beast_info_gain = info_gain
            beast_feature = i
    return beast_feature


data, _ = create_data()
feature = choose_bast_feature_to_split(data)
print(feature)


