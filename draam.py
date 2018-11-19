#! /usr/bin/env python3
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

word_vector_size = 300
padding = word_vector_size // 2


# Dynamic template used from here:
# https://github.com/jcjohnson/pytorch-examples#pytorch-control-flow--weight-sharing
class DynamicRAAM(torch.nn.Module):
    def __init__(self, D_in, H, D_out, max_sen_len):
        super(DynamicRAAM, self).__init__()
        self.input_size = max_sen_len
        self.depth_ingest = int(math.ceil(math.log(self.input_size, 2)))
        self.encoder = _encoder(D_in, H)
        self.decoder = _decoder(H, D_out)

    def _encoder(self):
        ingest = torch.autograd.Variable(torch.randn(None, input_size, word_vector_size + padding))
        new_sen_len = self.input_size
        for i in range(self.depth_ingest):
            R_array = []
            for j in range(0, new_sen_len, 2):
                if j == new_sen_len-1:
                    R_array.append(ingest[:,j])
                else:
                    temp = torch.cat([ingest[:,j], ingest[:,j+1]], axis=1)
                    R = self._build_encoder(temp)
                    R_array.append(R)
            ingest = torch.stack(R_array, axis=1)
            new_sen_len //= 2
        self.ingest = ingest

    def _decoder(self):
        egest = self.ingest
        new_sen_len = 1
        for i in range(self.depth_ingest):
            R_array = []
            for j in range(new_sen_len):
                R = self._build_decoder(egest[:,j])
                R_array.extend([R[:,:input_size//2], R[:,input_size//2:]])
            egest = torch.stack(R_array, axis=1)
            new_sen_len *=2
        egest = egest[:,0:sen_len,:]
        self.egest = egest

    def _build_encoder(self, inputs):
        size = inputs.shape[1].value
        encoded = self._make_fc(inputs, size, "E_first")
        encoded2 = self._make_fc(encoded, 3*size//4, "E_second")
        center = self._make_fc(encoded2, size/2, "center")
        return center

    def _build_decoder(self, inputs):
        size = inputs.shape[1].value
        decoded = self._make_fc(inputs, 3*size//2, "D_first")
        decoded2 = self._make_fc(decoded, 2*size, "D_second")
        return decoded2

    def _make_fc(self, input_tensor, output_size, name):
        input_size = input_tensor.get_shape().as_list()[1]
        # TODO: Change to use PyTorch
        W = tf.get_variable(name+"weights",[input_size, output_size],tf.float32,
                                                tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name+'bias',[output_size],tf.float32,tf.zeros_initializer())
        x = tf.nn.tanh(tf.matmul(input_tensor, W) + b)
        return x

    def forward(self, x):
        y_pred = 0
        return y_pred

    def backward(self, grad_output):
        grad_x = 0
        return grad_x

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('--lr', type=float, default=.001, help='learning rate')
    parser.add_argument('--training_file', type=str, default='train_data.txt', help='raw training data')
    parser.add_argument('--vec_file', type=str, default='data/wiki-news-300d-1M.vec', help='vector file')
    parser.add_argument('--vec_dim', type=int, default='300', help='dimensions of vector embeddings')
    parser.add_argument('--num-iter', type=int, default=10, help='run iterations')

    args = parser.parse_args()
    return args

def main(learning_rate):
    input_size = 2 * (word_vector_size + padding)
    num_epochs = 500
    max_sen_len = 32
    # PyTorch stuff

    print("Vector size: %d, with padding: %d" % (word_vector_size, padding))
    print("Learning rate: %f" % learning_rate)

    vectors = "data/wiki-news-300d-1M.vec"
    corpus = "data/austen.txt"

    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = DynamicRAAM(D_in, H, D_out, max_sen_len)

    sentence_dict = generate_samples(vectors, corpus, word_vector_size, padding)

    # use 4/5 of the sentences to train, and 1/5 to validate
    cut = (4 * len(sentence_dict.values())) // 5
    training_data = sentence_dict.values()[0:cut]
    testing_data = sentence_dict.values()[cut:]

    # Where the magic happens
    train(sess, train_step, np.array(training_data), loss, num_epochs, ingest, egest, original_sentence)
    test(sess, np.array(testing_data), loss, ingest, egest, original_sentence)

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def generate_samples(vectors, corpus, vec_size):
    word_dict = parse_word_vecs(vectors, vec_size)
    sentences = parse_data(corpus)
    sentence_dict = {}
    for sentence in sentences:
        res = get_vecs_from_sentence(sentence, word_dict)
        if res is not None:
            if res.shape[0] < 32:
                    padding = 32 - res.shape[0]
                    res = np.pad(res, [(0, padding), (0, 0)], mode='constant')
            elif res.shape[0] > 32:
                    res = res[0:32]
            sentence_dict[sentence] = res
    return sentence_dict

def get_vecs_from_sentence(sentence, word_dict):
    arr = []
    for word in re.findall(r"[\w]+|[^\s\w]", sentence):
        curr = word_dict.get(word)
        if curr is None:
            return None
        arr.append(curr)
    return np.array(arr)

def parse_word_vecs(vectors, vec_size):
    i = 1
    d = {}
    in_file = open(vec_path)

    next(in_file)
    print(d.keys())
    for line in in_file:
        parsed = line.split(' ', 1)
        vec = np.fromstring(parsed[1], dtype = float, count = vec_dim, sep = " ")
        d[parsed[0]] = vec
        i += 1

        if i % 100000 == 0:
           break

    in_file.close()

    return d

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

