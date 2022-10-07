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

from sklearn import datasets
from sklearn.model_selection import train_test_split

DataMeta = NamedTuple('DataMeta', [
    ('name', str), ('url', str)])


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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, "resources")
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

