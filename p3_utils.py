# -*- coding: utf-8 -*-
import csv
import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import tarfile
import wget

from typing import NamedTuple, Tuple, Dict
from pandas import DataFrame
from functools import partial

DataMeta = NamedTuple('DataMeta', [
    ('name', str), ('url', str)])

_IMDB = DataMeta(
    name='imdb',
    url='https://www.dropbox.com/s/l9pj9hy2ans3phi/imdb.tar.gz?dl=1')

_GLOVE = DataMeta(
    name='glove',
    url='https://www.dropbox.com/s/g5pkso42wq2ipti/glove.tar.gz?dl=1'
)

_EMOJI_MODEL = DataMeta(
    name='emoji_model_best',
    url='https://www.dropbox.com/s/93frkdrhs33ihfz/emoji_model_best.tar.gz?dl=1'
) 

def _download_data_if_needed(data_meta: DataMeta) -> str:
    """
    Download and extract dataset if needed
    return the path to the dataset
    """
    path = os.path.join('resources', data_meta.name)
    zip_path = path + '.tar.gz'

    if os.path.exists(path):
        print('data already available, skip downloading.')
    else:
        print('start downloading...')
        wget.download(data_meta.url, zip_path)

        print('start extracting compressed files...')
        with tarfile.open(zip_path) as tar:
            tar.extractall('resources')
        os.remove(zip_path)

        print('data files are now available at %s' % path)
    return path


def download_best_emoji_model() -> Tuple[str, str]:
    """
    Download pretrained emoji model, skip if already downloaded,
    return the path to the network json and weights
    """
    path = _download_data_if_needed(_EMOJI_MODEL)
    return os.path.join(path, 'network.json'), os.path.join(path, 'weights.h5')


def _get_train_test_df(data_meta: DataMeta) -> Tuple[DataFrame, DataFrame]:
    path = _download_data_if_needed(data_meta)
    train, test = tuple(
        pd.read_csv(os.path.join(path, file))
        for file in ['train.csv', 'test.csv'])
    print('{} loaded successfully.'.format(data_meta.name))
    return train, test


get_imdb_dataset = partial(_get_train_test_df, _IMDB)

def load_glove_vecs() -> Tuple[Dict[str, int], Dict[str, np.array]]:
    """
    Download (if necessary) and read GloVe. Two mappings are returned.
    1. Word to index, mapping from word to its index in vocabulary,
       needed for building Embedding layer in Keras)
    2. Word to vector, mapping from word to its vec
    """
    path = _download_data_if_needed(_GLOVE)
    path = os.path.join(path, 'glove.6B.50d.txt')
    print('loading glove... this may take a while...')
    with open(path, encoding='utf-8') as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line_components = line.strip().split()
            curr_word = line_components[0]
            words.add(curr_word)
            word_to_vec[curr_word] = np.array(
                line_components[1:], dtype=np.float64)
        i = 1
        word_to_index = {}
        for w in sorted(words):
            word_to_index[w] = i
            i = i + 1
    print('glove loaded successfully.')
    return word_to_index, word_to_vec


def load_emoji():
    def _read_csv(filename):
        phrase = []
        emoji = []

        with open(filename, encoding='utf-8') as csvDataFile:
            csvReader = csv.reader(csvDataFile)

            for row in csvReader:
                phrase.append(row[0])
                emoji.append(row[1])

        X = np.asarray(phrase)
        Y = np.asarray(emoji, dtype=int)

        return X, Y

    train_x, train_y = _read_csv('resources/emoji/train.csv')
    test_x, test_y = _read_csv('resources/emoji/test.csv')
    return train_x, test_x, train_y, test_y


def plot_history(his, metrics):
    """
    Given a history object returned from `fit` and the name of metrics,
    plot the curve of metrics against number of epochs.
    """
    for metric in metrics:
        plt.plot(his.history[metric], label=metric)
    plt.legend()


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


EMOJI_DICT = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
              "1": ":baseball:",
              "2": ":smile:",
              "3": ":disappointed:",
              "4": ":fork_and_knife:"}


def label_to_emoji(label):
    return emoji.emojize(EMOJI_DICT[str(label)], use_aliases=True)


def sentences_to_indices(X, word_to_index, max_len):
    """
    Return a array of indices of a given sentence.
    The sentence will be trimed/padded to max_len

    Args:
        X (np.ndarray): Input array of sentences, the shape is (m,)  where m is the number of sentences, each sentence is a str. 
        Example X: array(['Sentence 1', 'Setence 2'])
        word_to_index (dict[str->int]): map from a word to its index in vocabulary

    Return:
        indices (np.ndarray): the shape is (m, max_len) where m is the number of sentences
    """
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    return X_indices
