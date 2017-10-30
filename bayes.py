# _*_ coding: utf-8 _*_
# @Time     : 2017/10/19 15:49
# @Author    : Ligb
# @File     : bayes.py

"""
朴素贝叶斯分类器的实现
最终在一些邮件数据上实现垃圾邮件分类，验证性能
"""

from numpy import *
import re


def load_data_set():
    """
    构造几个用于单词检索的例句
    :return: 例句单词列表组成的列表，每个例句对应的标签，1代表脏字
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_labels = [0, 1, 0, 1, 0, 1]
    return posting_list, class_labels


def create_vocabulary_list(data_set):
    """
    创建词汇表
    :param data_set:将输入的句子中所有词汇组成词汇表
    :return: 所有词汇组成的列表
    """
    vocabulary_list = set([])
    for line in data_set:

        # 作并集
        vocabulary_list = vocabulary_list | set(line)
    return list(vocabulary_list)


def words_set_to_vectors(vocabulary_list, document_words):
    """
    将输入的一组词汇在词汇表中检索并返回一个向量（数字）
    :param vocabulary_list: 词汇表
    :param document_words: 输入的文档
    :return: 词汇向量
    """
    # 创建一个词汇表的词向量
    returned_vector = [0] * len(vocabulary_list)
    for word in document_words:

        # 单词出现在词汇表中的次数
        if word in vocabulary_list:
            returned_vector[vocabulary_list.index(word)] += 1
        else:
            print("the word:" + word + " is not in the vocabulary!")
    return returned_vector


def train_bayes_classifier(training_matrix, training_category):
    """
    训练朴素贝叶斯的先验概率和条件概率（二分类任务）
    条件概率的算法与西瓜书不太一致
    :param training_matrix: 所有的训练样本的词汇向量组成的列表
    :param training_category: 各个训练样本的标签
    :return: 两个类别的条件概率和 一类先验概率
    """
    num_train_docs = len(training_matrix)
    num_words = len(training_matrix[0])

    # 计算其中类标签为1语句的先验概率
    p_abusive = sum(training_category) / float(num_train_docs)

    # 拉普拉斯修正
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if training_category[i] == 1:
            p1_num += training_matrix[i]
            p1_denom += sum(training_matrix[i])
        else:
            p0_num += training_matrix[i]
            p0_denom += sum(training_matrix[i])

    # 计算两类的条件概率,用对数避免下溢
    p0_vect = log(p0_num / p0_denom)
    p1_vect = log(p1_num / p1_denom)
    return p0_vect, p1_vect, p_abusive


def bayes_classifier(test_vectors, p0_vect, p1_vect, p_abusive_class):
    """
    对待测样本进行分类
    :param test_vectors:待测样本词向量
    :param p0_vect: 0类词汇概率表
    :param p1_vect: 1类词汇概率表
    :param p_abusive_class: 1类样本的先验概率
    :return: 分类结果：1；0，
    """
    p_abusive = sum(test_vectors * p1_vect) + log(p_abusive_class)
    p_nonabusive = sum(test_vectors * p0_vect) + log(1 - p_abusive_class)
    if p_abusive > p_nonabusive:
        return 1
    else:
        return 0


def text_parse(long_string):
    """
    将字符串切以正则表达式切分，并过滤空字符和url
    :param long_string: 待切分字符串
    :return: 字符列表
    """
    list_tokens = re.split(r'\W*', long_string)
    return [tok.lower() for tok in list_tokens if len(tok) > 2]


def spam_test():
    """
    验证朴素贝叶斯分类器分类垃圾邮件时的错误率
    验证方法：交叉验证
    训练集大小：40；验证集大小：10
    :return: None
    """
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1,26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())

        # 所有邮件内容构成的列表
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)

    # 创建词汇表
    vocabulary_list = create_vocabulary_list(doc_list)
    training_set = list(range(50))
    test_set = []

    # 随机产生大小为10的验证集
    for i in range(10):
        random_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[random_index])
        del training_set[random_index]
    train_mat = []
    train_classes = []

    # 剩余邮件作为训练集
    for doc_index in training_set:
        train_mat.append(words_set_to_vectors(vocabulary_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])

    # 训练出概率大小
    p0_v, p1_v, p_spam = train_bayes_classifier(array(train_mat), array(train_classes))
    error_count = 0

    # 验证算法错误率
    for doc_index in test_set:
        word_vector = words_set_to_vectors(vocabulary_list, doc_list[doc_index])
        if bayes_classifier(word_vector, p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1

            # 输出分类错误的那一个邮件
            print(doc_list[doc_index])
    print('the error rate is:' + str(float(error_count) / len(test_set)))


def test_bayes():
    """
    测试bayes分类器的各函数
    :return: None
    """
    # 获取所有文档
    words, sentence_labels = load_data_set()

    # 创建包含所有词汇的词汇表
    vocabulary = create_vocabulary_list(words)

    # 构造训练样本的词向量列表
    train_mat = []
    for post_line in words:
        train_mat.append(words_set_to_vectors(vocabulary, post_line))
    p0, p1, pa = train_bayes_classifier(array(train_mat), array(sentence_labels))

    # 创建测试样例
    test_entry = ['love', 'my', 'dalmation']
    test_doc = array(words_set_to_vectors(vocabulary, test_entry))
    print('该语句被分类为：' + str(bayes_classifier(test_doc, p0, p1, pa)))
    test_entry = ['stupid', 'garbage']
    test_doc = array(words_set_to_vectors(vocabulary, test_entry))
    print('该语句被分类为:' + str(bayes_classifier(test_doc, p0, p1, pa)))


if __name__ == '__main__':
    test_bayes()
    spam_test()