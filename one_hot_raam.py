# Adapted from: https://github.com/sethRait/RAAM/blob/master/one-hot-iterative-raam.py
# Encodes and decodes pairs of one-hot vectors
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import argparse


def main(args):
    input_size = 52  # 2 letters, 26 bits each, 1-hot

    input1 = tf.placeholder(tf.float32, [None, input_size / 2])  # first letter
    input2 = tf.placeholder(tf.float32, [None, input_size / 2])  # second letter
    input_full = tf.concat([input1, input2], 1)  # not 2None x 6

    # layers
    encoded = make_fc(input_full, input_size, "encoder", 1)
    encoded2 = make_fc(encoded, 3 * input_size / 4, "second_hidden", 2)
    encoded3 = make_fc(encoded2, input_size / 2, "third_hidden", 2)
    decoded1 = make_fc(encoded3, 3 * input_size / 4, "decoder", 2)
    decoded2 = make_fc(decoded1, input_size, "second_decoder", 1)

    loss = tf.losses.mean_squared_error(labels=input_full, predictions=decoded2)
    train_step = tf.train.GradientDescentOptimizer(0.003).minimize(loss)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #training_set = generate_samples(input_size // 2)
    training_set = parse_sentences("test.txt")

    encs = produce_encodings("gram_file.txt")

    enc_training_data("test.txt", encs, 6)

    train(sess, train_step, training_set, loss, decoded2, input1, input2)
    test(sess, training_set, loss, decoded2, input1, input2)
    sess.close()


def make_fc(input_tensor, output_size, name, mode):
    W = tf.get_variable(name + "weights", [input_tensor.get_shape().as_list()[1], output_size], tf.float32,
                        tf.random_normal_initializer(stddev=0.1))
    b = tf.Variable(tf.zeros([output_size]))
    if mode == 1:
        x = tf.nn.sigmoid(tf.matmul(input_tensor, W) + b)
    else:
        x = tf.nn.relu(tf.matmul(input_tensor, W) + b)
    return x


# Creates a list of all pairwise combinations of 'size' distinct one-hot vectors
def generate_samples(n):
    a = generate_one_hots(n)
    idx = np.array([np.argmax(i) for i in a])
    putval = (idx[:, None] == np.arange(n)).astype(int)
    out = np.zeros((n, n, 2, n), dtype=int)
    out[:, :, 0, :] = putval[:, None, :]
    out[:, :, 1, :] = putval
    out.shape = (n ** 2, 2, -1)
    return out


# Creates a list of 'size' one-hot vectors
def generate_one_hots(size):
    out = (np.random.choice(size, size, replace=0)[:, None] == range(size)).astype(int)
    return list(map(list, out))


def chunks(l, n):
    # split l into n-sized chunks
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Returns a dictionary of word->encoding mappings
def produce_encodings(grammar_file):
    literals = []
    in_file = open(grammar_file, "r")

    for line in in_file:
        litre = re.compile("\'[a-z|A-Z|0-9]+\'")
        literals += ([re.sub("\'", '', item) for item in litre.findall(line)])

    in_file.close()

    total_len_enc = len(literals)
    enc_mapping = {}

    for iter, item in enumerate(literals):
        temp = ''

        for i in range(total_len_enc):
            if i == iter:
                temp += '1'
            else:
                temp += '0'

        enc_mapping[item] = temp

    return enc_mapping

def enc_training_data(corpus, encs, max_sen_len):
    in_file = open(corpus, "r")

    len_enc = len(list(encs.values())[0])

    for line in in_file:
        tokens = line.split()
        sen_encs = [encs[item] for item in tokens] + ['0' * len_enc] * (max_sen_len - len(tokens))

    in_file.close()

    return sen_encs

''' If the given vector (an np array) passes some terminal test, return the one-hot vector
 it most likely represents along with 'TRUE', else, return the given vector and 'FALSE'.'''


def good_or_bad(vec):
    affordance = 0.20  # How much the given value can differ from 0 or 1 to be considered terminal
    out_vec = np.array(vec)
    np.copyto(out_vec, vec)
    for x in np.nditer(out_vec, op_flags=['readwrite']):
        if 0 - affordance <= x <= affordance:
            x[...] = 0
        elif 1 - affordance <= x <= 1 + affordance:
            x[...] = 1
        else:
            return (False, vec)
    return (True, out_vec)

def train(sess, train_step, training_set, loss, decoded2, input1, input2):
    # Training loop
    for i in range(20000):
        # inputs = training_set[i % len(training_set)]
        x = np.array([training_set[j][0] for j in range(training_set.shape[0])])
        y = np.array([training_set[j][1] for j in range(training_set.shape[0])])
        random.shuffle(x)
        random.shuffle(y)

        #	_, my_loss, my_decoded, original = sess.run([train_step, loss, decoded2, input_full], feed_dict={input1:inputs[0]], input2:[inputs[1]]})
        _, my_loss, _, = sess.run([train_step, loss, decoded2], feed_dict={input1: x, input2: y})
        if i % 500 == 0:
            print("epoch: " + str(i))
            print("loss: " + str(my_loss))

def test(sess, training_set, loss, decoded2, input1, input2):
    # Testing loop
    for i in range(20000, 25000):
        x = np.array([training_set[j][0] for j in range(training_set.shape[0])])
        y = np.array([training_set[j][1] for j in range(training_set.shape[0])])
        random.shuffle(x)
        random.shuffle(y)

        my_loss, my_decoded, = sess.run([loss, decoded2], feed_dict={input1: x, input2: y})
        if i % 250 == 0:
            print("loss: " + str(my_loss))
            good, answer = good_or_bad(my_decoded)
            print("reconstructed? " + str(good))
            if good:
                print(answer)

def parse_args():
    parser = argparse.ArgumentParser(description='one_hot_raam.py')

    parser.add_argument('--training-file', type=str, default='data/austen.txt', help='raw training data')
    parser.add_argument('--verbose', action='store_true', help='verbose flag')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        print(args)

    for i in range(10):
        main(args)
        tf.reset_default_graph()
