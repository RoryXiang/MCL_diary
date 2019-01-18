from bayes import create_vocablist, text_pase, bag_words2vc_mn, train_nbo, classfy_nb
import operator
import random
import numpy as np
import feedparser


def calc_most_freq(vocablist, fulltext):
    """[summary]
        统计出现次数最多的前30个词
    Arguments:
        vocablist {[string list]} -- words list with every word is unique
        fulltext {[string list]} -- words list

    Returns:
        [type] -- [description]
    """
    freq_dic = {}
    for token in vocablist:
        freq_dic[token] = fulltext.count(token)
    sorted_frq = sorted(
        freq_dic.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_frq[:30]


def local_words(feed1, feed0):
    """[summary]

    [description]

    Arguments:
        feed1 {[type]} -- [description]
        feed0 {[type]} -- [description]
    """
    doclist = []
    classlist = []
    fulltext = []
    minlen = min(len(feed1["entries"]), len(feed0["entries"]))
    for i in range(minlen):
        wordlist = text_pase(feed1["entries"][i]["summary"])
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(1)
        wordlist = text_pase(feed0["entries"][i]["summary"])
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(0)
    vocablist = create_vocablist(doclist)
    top30words = calc_most_freq(vocablist, fulltext)
    for pairw in top30words:
        if pairw[0] in vocablist:
            vocablist.remove(pairw[0])
    trainingset = list(range(2 * minlen))
    test_set = []
    for i in range(20):
        random_index = int(random.uniform(0, len(trainingset)))
        test_set.append(trainingset[random_index])
        trainingset.pop(random_index)
    train_mat = []
    train_class = []
    for docindex in trainingset:
        train_mat.append(bag_words2vc_mn(vocblist, doclist[docindex]))
        train_class.append(classlist[docindex])
    p0v, p1v, pspam = train_nbo(np.array(train_mat), np.array(train_class))
    errorcount = 0
    for docindex in test_set:
        wordvector = bag_words2vc_mn(vocablist, doclist[docindex])
        if classfy_nb(np.array(wordvector), p0v, p1v, pspam) != classlist[docindex]:
            errorcount += 1
    print("the error rate is :", float(errorcount) / len(test_set))
    return vocablist, p0v, p1v


if __name__ == '__main__':
    a, b, c = local_words()
