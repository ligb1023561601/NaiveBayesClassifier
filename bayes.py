# _*_ coding: utf-8 _*_
# @Time     : 2017/10/19 15:49
# @Author    : Ligb
# @File     : bayes.py

from numpy import *


def load_data_set():
    """
    构造几个用于单词检索的例句
    :return: 例句单词列表组成的列表，每个例句对应的标签，1代表脏字
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting','stupid', 'worthless', 'garbage'],
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
        vocabulary_list = vocabulary_list | set(line)
    return list(vocabulary_list)


def words_set_to_vectors(vocabulary_list, document_words):
    """
    将输入的一组词汇在词汇表中检索并返回一个向量（数字）
    :param vocabulary_list: 词汇表
    :param document_words: 输入的文档
    :return: 词汇向量
    """
    returned_vector = [0] * len(vocabulary_list)
    for word in document_words:
        if word in vocabulary_list:
            returned_vector[vocabulary_list.index(word)] = 1
        else:
            print("the word:" + word + " is not in the vocabulary!")
    return returned_vector


def train_bayes_classifier(training_matrix, training_category):
    num_train_docs = len(training_matrix)
    num_words = len(training_matrix[0])
words, sentence_labels = load_data_set()
vocabulary = create_vocabulary_list(words)
print(words_set_to_vectors(vocabulary, ["hahaha", "I", "stupid"]))