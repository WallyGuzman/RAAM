import numpy as np
import torch.nn as nn
import argparse
import nltk

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('--lr', type=float, default=.001, help='learning rate')
    parser.add_argument('--training_file', type=str, default='train_data.txt', help='raw training data')

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
    create_vectors(sentences)