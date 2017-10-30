# _*_ coding: utf-8 _*_
# @Time     : 2017/10/30 10:19
# @Author    : Ligb
# @File     : advertisement_inclination.py

import feedparser
import operator
import random
from numpy import *

import bayes


def calc_most_freq(vocabulary_list, full_text):
    """
    计算词出现的频率
    :param vocabulary_list: 单词表
    :param full_text: 全文
    :return: 出现频率最高的30个词,调整此参数可改变错误率
    """
    freq_dict = {}
    for token in vocabulary_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    """
    用RSS源测试贝叶斯分类器
    :param feed1: RSS源1,ny
    :param feed0: RSS源2,sf
    :return: 词汇表，以及源1和源2的条件概率
    """
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):

        # 逐条获取RSS中的文本信息
        word_list = bayes.text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = bayes.text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    # 形成词汇表
    vocabulary_list = bayes.create_vocabulary_list(doc_list)

    # 去掉出现次数最高的30个词，调整去掉的词的数量可调整错误率
    top_30_words = calc_most_freq(vocabulary_list, full_text)
    for pair_w in top_30_words:
        if pair_w[0] in vocabulary_list:
            vocabulary_list.remove(pair_w[0])
    training_set = list(range(2 * min_len))
    test_set = []

    # 划分训练集和验证集
    for i in range(20):
        random_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[random_index])
        del training_set[random_index]
    train_mat = []
    train_classes = []

    # 以随机生成的训练集训练模型
    for doc_index in training_set:
        train_mat.append(bayes.words_set_to_vectors(vocabulary_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = bayes.train_bayes_classifier(array(train_mat), array(train_classes))
    error_count = 0

    # 检验模型性能
    for doc_index in test_set:
        word_vector = bayes.words_set_to_vectors(vocabulary_list, doc_list[doc_index])
        if bayes.bayes_classifier(word_vector, p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print('the error rate is:' + str(float(error_count) / len(test_set)))
    return vocabulary_list, p0_v, p1_v


def get_top_words(ny, sf):
    """
    查看出现概率最大的词
    :param ny: RSS源1
    :param sf: RSS源2
    :return: None
    """
    vocab_list, p_sf, p_ny = local_words(ny, sf)
    top_ny = []
    top_sf = []

    # 设置一个阈值可决定显示的词的数量
    for i in range(len(p_sf)):
        if p_sf[i] > -4.5:
            top_sf.append((vocab_list[i], p_sf[i]))
        if p_ny[i] > -4.5:
            top_ny.append((vocab_list[i], p_ny[i]))
    sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sorted_sf:
        print(item)
    sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sorted_ny:
        print(item)


if __name__ == '__main__':
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    get_top_words(ny, sf)

