# Adapted from here:
# https://github.com/sethRait/RAAM/blob/master/draam.py

from __future__ import division
import numpy as np
import torch.nn as nn
import argparse
import nltk
import numpy as np
import re
import math
from scipy import spatial

def main(learning_rate):
    word_vector_size = 300
    input_size = 2 * word_vector_size
    num_epochs = 500
    # PyTorch stuff

    vectors = "data/wiki-news-300d-1M.vec"
    corpus = "data/austen.txt"


def build_encoder(inputs):
    size = inputs.shape[1].value
    # PyTorch stuff

def build_decoder(inputs):
    size = inputs.shape[1].value
    # PyTorch stuff

def make_fc(input_tensor, output_size, name):
    # PyTorch stuff
    input_size = input_tensor.get_shape().as_list()[1]

# TODO: Change this so there's no padding?
def generate_samples(vectors, corpus, vec_size, pad):
    word_dict = parse_word_vecs(vectors, vec_size, pad)
    sentences = parse_sentences(corpus)
    sentence_dict = {}
    for sentence in sentences:
        res = get_vecs_from_sentence(sentence, word)
        if res is not None:
            # if res.shape[0] < 32:
            #     padding = 32 - res.shape[0]
            #     res = np.pad(res, [(0, padding), (0, 0)], mode='constant')
            # elif res.shape[0] > 32:
            #     res = res[0:32]
            sentence_dict[sentence] = res
    return sentence_dict

def get_vecs_from_sentence(sentence, word_dict):
    arr = []
    for word in re.findall(r"[\w]+|[^\s\w]", sentence):
        cur = word_dict.get(word.lower())
        if cur is None:
            return None
        arr.append(cur)
    return np.array(arr)

def parse_word_vecs(vectors, vec_size, pad):
    # FastText stuff
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('--lr', type=float, default=.001, help='learning rate')
    parser.add_argument('--training_file', type=str, default='train_data.txt', help='raw training data')
    parser.add_argument('--num-iter', type=int, default=10, help='run iterations')

    args = parser.parse_args()
    return args

def parse_data(data):
    in_file = open(data)

    nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = nltk.sent_tokenize(in_file.read())

    in_file.close()
    return sentences

def parse_sentences(corpus):
    with open(corpus) as fp:
        nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = nltk.sent_tokenize(fp.read().decode('utf-8'))
    return sentences

def train():
    pass

def test():
    pass

if __name__ == "__main__":
    nltk.download('punkt')
    args = parse_args()
    learning_rate = args.learning_rate

    for _ in range(args.num_iter):
        main(learning_rate)
        learning_rate /= 2

