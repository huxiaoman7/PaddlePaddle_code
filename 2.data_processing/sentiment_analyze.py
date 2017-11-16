#coding:utf-8
'''
Created by huxiaoman 2017.11.15
sentiment_analyze.py: create reader to convert text data to train and test
'''
import os

def train_reader(data_dir, word_dict, label_dict):
    """
   创建训练数据reader
    :param data_dir: 数据地址.
    :type data_dir: str
    :param word_dict: 词典地址,
        词典里必须有 "UNK" .
    :type word_dict:python dict
    :param label_dict: label 字典的地址
    :type label_dict: Python dict
    """

    def reader():
        UNK_ID = word_dict["<UNK>"]
        word_col = 1
        lbl_col = 0

        for file_name in os.listdir(data_dir):
            with open(os.path.join(data_dir, file_name), "r") as f:
                for line in f:
                    line_split = line.strip().split("\t")
                    word_ids = [
                        word_dict.get(w, UNK_ID)
                        for w in line_split[word_col].split()
                    ]
                    yield word_ids, label_dict[line_split[lbl_col]]

    return reader


def test_reader(data_dir, word_dict):
    """
    创建测试数据reader
    :param data_dir: 数据地址.
    :type data_dir: str
    :param word_dict: 词典地址,
        词典里必须有 "UNK" .
    :type word_dict:python dict
    """

    def reader():
        UNK_ID = word_dict["<UNK>"]
        word_col = 1

        for file_name in os.listdir(data_dir):
            with open(os.path.join(data_dir, file_name), "r") as f:
                for line in f:
                    line_split = line.strip().split("\t")
                    if len(line_split) < word_col: continue
                    word_ids = [
                        word_dict.get(w, UNK_ID)
                        for w in line_split[word_col].split()
                    ]
                    yield word_ids, line_split[word_col]

    return reader
