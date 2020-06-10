# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split

import csv
import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import tarfile
import wget

from typing import NamedTuple, Tuple, Dict

DataMeta = NamedTuple('DataMeta', [
    ('name', str), ('url', str)])

_GLOVE = DataMeta(
    name='glove',
    url='https://www.dropbox.com/s/g5pkso42wq2ipti/glove.tar.gz?dl=1'
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


def load_moon():
    '''Load moon dataset from sklearn'''
    X, Y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=1234)
    Y = Y.reshape(Y.shape[0], 1)
    return train_test_split(X, Y, test_size=0.2, random_state=1234)


def plot_decision_boundary(model, X, y):
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = 0.01
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, h),
        np.arange(ymin, ymax, h)
    )

    logits = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.where(logits >= 0.5, 1, 0).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.T[0], cmap=plt.cm.Spectral)


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


def sentence_to_avg(sentence, word_to_vec_map):
    import numpy as np

    # Convert a sentence string into the average of word vector (dim = 50)
    words = sentence.lower().split()

    avg = np.zeros((50,))
    cnt = 0
    for w in words:
        avg += word_to_vec_map.get(w, np.zeros((50,)))
        cnt += 1
    avg = avg / cnt

    return avg
