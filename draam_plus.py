# Adapted from: https://github.com/sethRait/RAAM/blob/master/draam.py
# Recursiely encodes and decodes pairs of word vectors
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import re
import sys
import argparse
import math
from scipy import spatial
from find_nn import find_nn

SEED = 42

def main(args):
    word_vector_size = args.vec_dim
    padding = word_vector_size // 2
    input_size = 2 * (word_vector_size + padding)
    num_epochs = 500
    sen_len = 32
    hidden_size = args.hidden_size
    learning_rate = args.lr
    freq_report = args.freq_report
    report_test = args.report_test
    word_dim_size = word_vector_size + padding

    print("Vector size: %d, with padding: %d" % (word_vector_size, padding))
    print("Learning rate: %f" % learning_rate)

    vectors = args.vec_file # File of word vectors
    corpus = args.training_file
    #test_corpus = args.test_file

    original_sentence = tf.placeholder(tf.float32, [None, sen_len, word_vector_size + padding])
    ingest = original_sentence

    keep_prob = tf.placeholder(tf.float32)

    # ingest
    depth_ingest = int(math.ceil(math.log(sen_len, 2)))
    new_sen_len = sen_len
    with tf.name_scope('encoder'):
        for i in range(depth_ingest):
            with tf.name_scope(str(i)):
                R_array = []
                for j in range(0, new_sen_len, 2):
                    if j == new_sen_len - 1:
                        R_array.append(ingest[:, j])
                    else:
                        temp = tf.concat([ingest[:, j], ingest[:, j + 1]], axis=1)
                        R = build_encoder(temp, hidden_size, args, keep_prob)
                        R_array.append(R)
                ingest = tf.stack(R_array, axis=1)
                new_sen_len //= 2

    # egest
    egest = ingest
    new_sen_len = 1
    with tf.name_scope('decoder'):
        for i in range(depth_ingest):
            with tf.name_scope(str(i)):
                R_array = []
                for j in range(new_sen_len):
                    R = build_decoder(egest[:, j], args, keep_prob)
                    R_array.extend([R[:, :input_size // 2], R[:, input_size // 2:]])
                egest = tf.stack(R_array, axis=1)
                new_sen_len *= 2
        egest = egest[:, 0:sen_len, :]

    loss = tf.losses.mean_squared_error(labels=original_sentence, predictions=egest)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("checkpoints/", sess.graph)
    # print '*'*80
    #    for i in tf.trainable_variables():
    #        print(i)
    # print '*'*80

    sentence_dict, word_dict = generate_samples(vectors, corpus, word_vector_size, padding)
    #test_sentence_dict = generate_samples(vectors, test_corpus, word_vector_size, padding)

    # use 4/5 of the sentences to train, and 1/5 to validate
    cut = (4 * len(sentence_dict.values())) // 5
    training_data = list(sentence_dict.values())[0:cut]
    testing_data = list(sentence_dict.values())[cut:]
    #training_data = list(sentence_dict.values())
    #testing_data = list(test_sentence_dict.values())
    word_dict['<NULL>'] = [0] * word_dim_size
    gold_sentences_list = [item.split() + (['<NULL>'] * (32 - len(item.split()))) for item in list(sentence_dict.keys())]


    # Where the magic happens
    train(sess, train_step, np.array(training_data), gold_sentences_list[0:cut], word_dict, freq_report, loss, num_epochs, ingest, egest, original_sentence, word_dim_size, args, keep_prob)
    test(sess, np.array(testing_data), gold_sentences_list[cut:], word_dict, report_test, loss, ingest, egest, original_sentence, word_dim_size, args, keep_prob)
    sess.close()


def build_encoder(inputs, hidden_size, args, keep_prob):
    size = inputs.shape[1].value
    with tf.name_scope('encoder') as scope:
        encoded = make_fc(inputs, size, "E_first", args)
        encoded2 = make_fc(encoded, 3 * size // 4, "E_second", args)
    with tf.name_scope('center') as scope:
        center1 = make_fc(encoded2, size / 2, "center", args)
        drop_out = tf.nn.dropout(center1, keep_prob, seed=SEED)
        if args.extra_hidden:
            center2 = make_fc(drop_out, size / 2, "center2", args)
            drop_out2 = tf.nn.dropout(center2, keep_prob, seed=SEED)
            return drop_out2
    return drop_out


def build_decoder(inputs, args, keep_prob):
    size = inputs.shape[1].value
    with tf.name_scope('decoder') as scope:
        decoded = make_fc(inputs, 3 * size // 2, "D_first", args)
        decoded2 = make_fc(decoded, 2 * size, "D_second", args)
    return decoded2


def make_fc(input_tensor, output_size, name, args):
    input_size = input_tensor.get_shape().as_list()[1]
    with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
        W = tf.get_variable(name + "weights", [input_size, output_size], tf.float32,
                            tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name + 'bias', [output_size], tf.float32, tf.zeros_initializer())
        if args.activation == "tanh":
            x = tf.nn.tanh(tf.matmul(input_tensor, W) + b)
        elif args.activation == "relu":
            x = tf.nn.relu(tf.matmul(input_tensor, W) + b)
    return x


# Returns a dictionary of sentances and a list of their vector representation
def generate_samples(vectors, corpus, vec_size, pad):
    word_dict = parse_word_vecs(vectors, vec_size, pad)
    sentences = parse_sentences(corpus)
    sentence_dict = {}
    for sentence in sentences:
        res = get_vecs_from_sentence(sentence, word_dict)
        if res is not None:
            # Now we need the sentence to be length 30 (sentence.shape[0] == 30)
            if res.shape[0] < 32:
                padding = 32 - res.shape[0]
                res = np.pad(res, [(0, padding), (0, 0)], mode='constant')
            elif res.shape[0] > 32:
                res = res[0:32]
            sentence_dict[sentence] = res
    return sentence_dict, word_dict


# Returns an np array of vectors representing the words of the given sentence
def get_vecs_from_sentence(sentence, word_dict):
    arr = []
    for word in re.findall(r"[\w]+|[^\s\w]", sentence):  # Each punctuation mark should be its own vector
        cur = word_dict.get(word.lower())
        if cur is None:
            return None
        arr.append(cur)
    return np.array(arr)


# Parses the file containing vector representations of words
def parse_word_vecs(vectors, vec_size, pad):
    i = 1
    dictionary = {}
    with open(vectors) as fp:
        next(fp)  # skip header
        for line in fp:
            parsed = line.lower().split(' ', 1)
            vec = np.fromstring(parsed[1], dtype=float, count=vec_size, sep=" ")
            dictionary[parsed[0]] = np.pad(vec, (0, pad), 'constant')  # right pad the vector with 0
            i += 1
            if i % 100000 == 0:  # Only use the first 100,000 words
                break
    return dictionary


# Parses the file containing the training and testing sentences
def parse_sentences(corpus):
    with open(corpus) as fp:
        # nltk.data.load('tokenizers/punkt/english.pickle')
        # sentences = nltk.sent_tokenize(fp.read())
        sentences = [line.split("\n")[0] for line in fp]
    return sentences


def train(sess, optimizer, data, gold_sentences, word_dict, freq_report, loss, num_epochs, ingest, egest, orig, word_dim_size, args, keep_prob):
    print("Shape is: ")
    print(data.shape)
    for i in range(num_epochs):
        _, train_loss, encoded, decoded = sess.run([optimizer, loss, ingest, egest], feed_dict={orig: data, keep_prob: args.keep_prob})
        if i % 25 == 0:
            print("Epoch: " + str(i))
            print("Loss: " + str(train_loss))
        if freq_report != -1 and freq_report != -2:
            if i % freq_report == 0:
                find_nn(decoded, gold_sentences, word_dict, word_dim_size)
        elif freq_report == -2:
            continue
        else:
            if (i + 1) % num_epochs == 0:
                find_nn(decoded, gold_sentences, word_dict, word_dim_size)


# Testing loop
def test(sess, data, gold_sentences, word_dict, report_test, loss, ingest, egest, orig, word_dim_size, args, keep_prob):
    test_loss, _encoded, decoded = sess.run([loss, ingest, egest], feed_dict={orig: data, keep_prob: 1.0})
    check_data = data[0]
    check_output = decoded[0]
    zipped = zip(check_data, check_output)
    result = 1 - spatial.distance.cosine(check_data[0], check_output[0])

    if report_test:
        np.save(args.word_file, decoded)
        #find_nn(decoded, gold_sentences, word_dict, word_dim_size)

    print("cosine: " + str(result))
    print("Validation loss: " + str(test_loss))

def parse_args():
    parser = argparse.ArgumentParser(description='draam_plus.py')

    parser.add_argument('--lr', type=float, default=.0001, help='learning rate')
    parser.add_argument('--training-file', type=str, default='data/austen.txt', help='raw training data')
    #parser.add_argument('--test-file', type=str, default='data/austen.txt', help='raw test data')
    parser.add_argument('--vec-file', type=str, default='/home/ubuntu/NN/RAAM/data/wiki-news-300d-1M.vec', help='word vector file')
    parser.add_argument('--vec-dim', type=int, default=300, help='word vector dimension')
    parser.add_argument('--verbose', action='store_true', help='verbose flag')
    parser.add_argument('--hidden-size', type=int, default=300, help='size of hidden layer')
    parser.add_argument('--keep-prob', type=float, default=0.5, help='use dropout for each layer')
    parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh', help='activation function')
    parser.add_argument('--extra-hidden', action='store_true', help='second hidden layer')
    parser.add_argument('--freq-report', type=int, default=-2, help='frequency of word reports')
    parser.add_argument('--report-test', action='store_true', help='word reports toggle for testing')
    parser.add_argument('--word-file', type=str, default='draam_plus_word_vecs.npy', help='output word vector file')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        print(args)

    main(args)
