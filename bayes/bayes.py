import numpy as np
import re
import random


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
    :param vocab_lsit: 词汇表
    :param input_set: 文档词汇
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
    :param train_matrix: 训练的文档矩阵
    :param train_category: 训练的文档类别向量
    :return:
    """
    print(train_matrix)
    num_traindocs = len(train_matrix)
    num_words = len(train_matrix[0])
    pa_busive = np.sum(train_category) / float(num_traindocs)  # 侮辱性文章的比例
    # =====================================================
    # p0_num = p1_num = np.zeros(num_words)
    # p0_denom = p1_denom = 0
    # 避免0次出现的影响======================================
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
    # ======================================================
    # p1_vec = p1_num / p1_denom
    # p0_vec = p0_num / p1_denom
    # 避免溢出问题===========================================
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, pa_busive


if __name__ == '__main__':
    list_posts, list_class = load_data()
    myvocb_list = create_vocablist(list_posts)
    train_mat = []
    for postindoc in list_posts:
        train_mat.append(set_of_words2vec(myvocb_list, postindoc))
    p0v, p1v, pab = train_nbo(train_mat, list_class)
    print(pab)


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

    # =========================================================================
    test_entry1 = ["stupid", "garbage"]
    this_doc1 = np.array(set_of_words2vec(myvocb_list, test_entry1))
    print(f"{test_entry1} classify as :", classfy_nb(this_doc1, p0v, p1v, pab))


# if __name__ == '__main__':
#     testing_nb()


# 朴素贝叶斯词袋模型
def bag_words2vc_mn(vocablist, inputset):
    """[summary]
        构建次贷模型，就是词概率    
    Arguments:
        vocablist {[type]} -- [description]
        inputset {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    return_vec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            return_vec[vocablist.index(word)] += 1  # 和上面的区别，为什么
    return return_vec


# 过滤垃圾邮件
def text_pase(bigstring):
    """[summary]
        将字符串分词成单个的单词，
    Arguments:
        bigstring {[string]} -- [description]
    """
    list_of_tokens = re.split(r"\W*", bigstring)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    doc_list = []  # 文档词汇表 [[],[]]
    class_list = []
    full_text = []  # 词汇向量（包含重复）
    # 读取所有邮件的词汇，构建词汇表
    for i in range(1, 26):
        word_list_ = text_pase(
            open(f"./email/spam/{i}.txt", encoding="utf-8", errors="ignore").read())
        doc_list.append(word_list_)
        full_text.extend(word_list_)
        class_list.append(1)
        word_list_ = text_pase(
            open(f"./email/ham/{i}.txt", encoding="utf-8", errors="ignore").read())
        doc_list.append(word_list_)
        full_text.extend(word_list_)
        class_list.append(0)

    vocab_list = create_vocablist(doc_list)  # 词汇向量（所有的不重复词）
    training_set = list(range(50))
    test_set = []
    # 选出10个测试集，其余40做训练集
    for k in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        training_set.pop(rand_index)

    train_mat = []
    train_class = []
    for doc_index in training_set:
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p0v, p1v, pSpam = train_nbo(np.array(train_mat), np.array(train_class))
    errorcount = 0
    for docindex in test_set:
        word_vector = set_of_words2vec(vocab_list, doc_list[docindex])
        if classfy_nb(np.array(word_vector), p0v, p1v, pSpam) != class_list[docindex]:
            errorcount += 1
    print("the error rate is : ", float(errorcount) / len(test_set))


# if __name__ == '__main__':
#     spam_test()
#
