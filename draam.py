import numpy as np
import torch.nn as nn
import argparse
import nltk

VEC_DIM = 300

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('--lr', type=float, default=.001, help='learning rate')
    parser.add_argument('--training_file', type=str, default='train_data.txt', help='raw training data')
    parser.add_argument('--vec_file', type=str, default='data/wiki-news-300d-1M.vec', help='vector file')

    args = parser.parse_args()
    return args

def get_vecs_from_sentence(sentence, word_dict):
    temp = []

    for word in re.findall(r"[\w]+|[^\s\w]", sentence):
        curr = word_dict.get(word.lower())

        if curr is None:
            return None
        temp.append(curr)
    return np.array(temp)

def parse_word_vecs(vec_path, vec_dim):
    i = 1
    dict = {}
    in_file = open(vec_path)

    next(in_file)
    print(dict.keys())
    for line in in_file:
        parsed = line.split(' ', 1)
        vec = np.fromstring(parsed[1], dtype = float, count = vec_dim, sep = " ")
        dict[parsed[0]] = vec
        i += 1

        if i % 100000 == 0:
            break

    in_file.close()

    return dict

def parse_data(data):
    in_file = open(data)

    nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = nltk.sent_tokenize(in_file.read())

    in_file.close()
    return sentences

if __name__ == "__main__":
    nltk.download('punkt')
    args = parse_args()

    sentences = parse_data(args.training_file)
    dict = parse_word_vecs(args.vec_file, VEC_DIM)
    print(len(list(dict.keys())))