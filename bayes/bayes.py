import numpy as np


def load_data():
    post_list = [
        ["my", "dog", "has", "flea", "problems", "help", "please"],
        ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
        ["stop", "posting", "stupid", "worthless", "garbage"],
        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
        ["quit", "buying", "worthless", "dog", "food", "stupid"]
    ]
    class_vec = [0, 1, 0, 1, 0, 1]
    return post_list, class_vec


def create_vocablist(data_set):
    """
    词去重，统计所有出现过的词
    :param data_set:
    :return:
    """
    vocab_set = set()
    for document in data_set:
        vocab_set = vocab_set | set(document)  # 求两个集合的并集
    return list(vocab_set)


# 词袋模型
def set_of_words2vec(vocab_lsit, input_set):
    """
    将词文本转换成0,1的数字向量
    :param vocab_lsit:
    :param input_set:
    :return:
    """
    return_vec = [0] * len(vocab_lsit)
    for word in input_set:
        if word in vocab_lsit:
            return_vec[vocab_lsit.index(word)] = 1
        else:
            print(f"the word: {word} is not in my Vocabulary!")
    return return_vec


# if __name__ == '__main__':
#     list_posts, list_class = load_data()
#     myvocb_list = create_vocablist(list_posts)
#     print(myvocb_list)
#     p = set_of_words2vec(myvocb_list, list_posts[0])
#     print(p)


def train_nbo(train_matrix, train_category):
    """
    朴素贝叶斯分离器函数
    :param train_matrix:
    :param train_category:
    :return:
    """
    num_traindocs = len(train_matrix)
    num_words = len(train_matrix[0])
    pa_busive = np.sum(train_category) / float(num_traindocs)
    # ------------------------------------------------------
    # p0_num = p1_num = np.zeros(num_words)
    # p0_denom = p1_denom = 0
    # 避免0次出现的影响-------------------------------------
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = p1_denom = 2
    for i in range(num_traindocs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    # -------------------------------------------------------
    # p1_vec = p1_num / p1_denom
    # p0_vec = p0_num / p1_denom
    # 避免溢出问题-------------------------------------------
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, pa_busive


# if __name__ == '__main__':
#     list_posts, list_class = load_data()
#     myvocb_list = create_vocablist(list_posts)
#     train_mat = []
#     for postindoc in list_posts:
#         train_mat.append(set_of_words2vec(myvocb_list, postindoc))
#     p0v, p1v, pab = train_nbo(train_mat, list_class)
#     print(pab)


def classfy_nb(vec2classify, p0vec, p1vec, pclass1):
    """
    判断文档类型
    :param vec2classify:
    :param p0vec:
    :param p1vec:
    :param pclass1:
    :return:
    """
    p1 = np.sum(vec2classify * p1vec) + np.log(pclass1)
    p0 = np.sum(vec2classify * p0vec) + np.log(1 - pclass1)
    print("////", p1, p0)
    if p1 > p0:
        return 1
    return 0


def testing_nb():
    list_posts, list_class = load_data()
    myvocb_list = create_vocablist(list_posts)
    train_mat = []
    for postindoc in list_posts:
        train_mat.append(set_of_words2vec(myvocb_list, postindoc))
    p0v, p1v, pab = train_nbo(np.array(train_mat), np.array(list_class))
    # print(">>>>>>", p0v, p1v)
    print(pab)

    test_entry = ["love", "my", "dalmation"]
    this_doc = np.array(set_of_words2vec(myvocb_list, test_entry))
    print(f"{test_entry} classify as :", classfy_nb(this_doc, p0v, p1v, pab))

    # -------------------------------------------------------------------
    test_entry1 = ["stupid", "garbage"]
    this_doc1 = np.array(set_of_words2vec(myvocb_list, test_entry1))
    print(f"{test_entry1} classify as :", classfy_nb(this_doc1, p0v, p1v, pab))


# if __name__ == '__main__':
#     testing_nb()


# 朴素贝叶斯词袋模型
def bag_words2vc_mn(vocablist, inputset):
    return_vec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            return_vec[vocablist.index(word)] += 1  # 和上面的区别，为什么
    return return_vec

