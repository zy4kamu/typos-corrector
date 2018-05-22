import argparse, os
import tensorflow as tf
import numpy as np
from convertions import NUM_SYMBOLS, numpy_to_string, string_to_numpy, SPACE_INT, int_to_char
from dataset_reader import DataSetReader
from prefix_tree import PrefixTree

command      = None
input_folder = None
message_size = None
batch_size   = None
model_file   = None

train_start_batch = 0
train_end_batch = 22500
test_start_batch = 24000
test_end_batch = 24500

def train_network():
    with tf.device("/device:GPU:0"):
        # input labels
        clean_tokens = tf.placeholder(tf.int32, [None, message_size], name='clean_tokens')                                        
        clean_one_hot_embedding = tf.one_hot(clean_tokens, NUM_SYMBOLS)                                            

        # predictions
        embedding_size = NUM_SYMBOLS
        contaminated_tokens = tf.placeholder(tf.int32, [None, message_size], name='contaminated_tokens')                                 
        char_embedding_matrix = tf.Variable(tf.random_uniform([NUM_SYMBOLS, embedding_size], -1.0, 1.0))
        contaminated_embedding = tf.nn.embedding_lookup(char_embedding_matrix, contaminated_tokens)              
        contaminated_embedding = tf.reshape(contaminated_embedding, (-1, message_size * embedding_size))   

        logits = []
        for _ in range(message_size):
            hidden1 = 200
            weights1 = tf.Variable(tf.truncated_normal([message_size * embedding_size, hidden1]))
            biases1  = tf.Variable(tf.truncated_normal([hidden1]))
            layer1 = tf.sigmoid(tf.matmul(contaminated_embedding, weights1) + biases1)                                 
            weights = tf.Variable(tf.truncated_normal([hidden1, NUM_SYMBOLS])) 
            biases  = tf.Variable(tf.truncated_normal([NUM_SYMBOLS])) 
            logit = tf.matmul(layer1, weights) + biases
            logits.append(tf.identity(logit, 'logit_{}'.format(_)))

        # loss and optimizer
        total_loss = None
        for i in range(message_size):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits[i], labels=clean_one_hot_embedding[:, i, :])
            loss = tf.reduce_sum(loss)
            if total_loss is None: total_loss = loss
            else: total_loss += loss
    optimizer = tf.train.AdamOptimizer().minimize(total_loss)

    # train network
    initializer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initializer)
        counter = 0
        train_reader = DataSetReader(input_folder, message_size, batch_size, train_start_batch, train_end_batch) 
        train_num_correct = 0
        train_num_letters = 0
        for clean, contaminated in train_reader:
            # update gradient
            _, predictions, l = sess.run([optimizer, logits, total_loss], 
                    feed_dict={clean_tokens:clean, contaminated_tokens:contaminated})
            # update statistics on train
            train_num_correct = 0
            train_num_letters = batch_size * message_size
            for i in range(message_size):
                train_num_correct += np.sum((np.argmax(predictions[i], axis=1) == clean[:, i]))
            counter += 1
            print '\r', counter,
            if counter == 2500:
                print '\r',
                counter = 0
                # print current model
                saver = tf.train.Saver()
                saver.save(sess, model_file)
                # print current state
                print 'epoch = {}'.format(train_reader.epoch)
                print 'train: {} correct of {}; accuracy = {}'.format(train_num_correct, train_num_letters, 
                                                                      float(train_num_correct) / float(train_num_letters))
                test_network(sess, logits, contaminated_tokens)
                print ''

def test_network(sess=None, logits=None, contaminated_tokens=None):
    if sess is None:
        sess = tf.Session()
        saver = tf.train.import_meta_graph(model_file + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file + '.meta')))
        graph = tf.get_default_graph()

        clean_tokens = graph.get_tensor_by_name("clean_tokens:0")
        contaminated_tokens = graph.get_tensor_by_name("contaminated_tokens:0")
        logits = [graph.get_tensor_by_name('logit_{}:0'.format(i)) for i in range(message_size)]

    test_reader = DataSetReader(input_folder, message_size, batch_size, test_start_batch, test_end_batch) 
    test_num_correct = 0
    test_num_letters = 0
    dummy_num_correct = 0
    for clean, contaminated in test_reader:
        if test_reader.epoch > 0: break 
        predictions, = sess.run([logits], feed_dict={ contaminated_tokens:contaminated })
        test_num_letters += batch_size * message_size
        for i in range(message_size):
            test_num_correct += np.sum((np.argmax(predictions[i], axis=1) == clean[:, i]))
            dummy_num_correct += np.sum(contaminated[:, i] == clean[:, i])
    print 'test: {} correct of {}; accuracy = {}'.format(test_num_correct, test_num_letters, 
                                                         float(test_num_correct) / float(test_num_letters))
    print 'dummy: {} correct of {}; accuracy = {}'.format(dummy_num_correct, test_num_letters, 
                                                          float(dummy_num_correct) / float(test_num_letters))


def play(prefix_tree_file):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_file + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file + '.meta')))
        graph = tf.get_default_graph()

        clean_tokens = graph.get_tensor_by_name("clean_tokens:0")
        contaminated_tokens = graph.get_tensor_by_name("contaminated_tokens:0")
        logits = [graph.get_tensor_by_name('logit_{}:0'.format(i)) for i in range(message_size)]

        while True:
            token = raw_input("Input something: ")
            numpy_token = np.ones(message_size, dtype=np.int32) * SPACE_INT
            numpy_token[0:len(token)] = string_to_numpy(token)
            numpy_token = numpy_token.reshape((-1, message_size))
            predictions, = sess.run([logits], feed_dict={contaminated_tokens:numpy_token})
            best_chars = [np.argmax(predictions[i], axis=1) for i in range(message_size)]
            print 'lazy best token: ', numpy_to_string(best_chars)
            print ''
            viterbi(predictions, prefix_tree_file, token)

def viterbi(predictions, prefix_tree_file, token, num_hypothesis=15):
    class Node(object):
        def __init__(self):
            self.choices = {}
        def add(self, prefix, value):
            self.choices[prefix] = value
        def reduce(self, max_num):
            buffer = [(prefix, value) for (prefix, value) in self.choices.iteritems()]
            buffer = sorted(buffer, key=lambda (x, y): -y)
            if len(buffer) > max_num:
                self.choices = {}
                for i in range(max_num):
                    self.choices[buffer[i][0]] = buffer[i][1]

    tree = PrefixTree(prefix_tree_file)
    predictions = [p[0] for p in predictions]
    num_choices = len(predictions[0])
    message_size = len(predictions)
    grid = [[Node() for j in range(num_choices)] for i in range(message_size)]

    # first layer
    for j in range(num_choices):
        grid[0][j].add(int_to_char(j), predictions[0][j])

    # evaluate intermideate layers
    for i in range(1, message_size):
        for j in range(num_choices):
            letter = int_to_char(j)
            for node in grid[i - 1]:
                for prefix, value in node.choices.iteritems():
                    new_prefix = (prefix + letter).strip()
                    new_value = value + predictions[i][j]
                    if tree.match(new_prefix) == len(new_prefix):
                        grid[i][j].add(new_prefix, new_value)
            grid[i][j].reduce(num_hypothesis)

    # evaluate last layer
    last_node = Node()
    for node in grid[-1]:
        for prefix, value in node.choices.iteritems():
            if tree.match(prefix + '$') == len(prefix) + 1:
                last_node.add(prefix, value)
    last_node.reduce(num_hypothesis)

    # print results
    buffer = [(prefix, levenshtein(prefix, token), value) for (prefix, value) in last_node.choices.iteritems()]
    buffer = sorted(buffer, key=lambda (x, y, z): float(100000 * y) - z)
    for k, l, v in buffer:
        print k, l, v

def levenshtein(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test neural network for typos correction')
    parser.add_argument('-c', '--command',            type=str, help='command to process',            required=True, choices=['train', 'test', 'play'])
    parser.add_argument('-i', '--input-folder',       type=str, help='folder with binary batches',    default='../dataset')
    parser.add_argument('-m', '--message-size',       type=int, help='length of each token in batch', default=10)
    parser.add_argument('-b', '--batch-size',         type=int, help='number of tokens in batch',     default=500)
    parser.add_argument('-f', '--model-file',         type=str, help='file with binary model',        default=None)
    parser.add_argument('-t', '--prefix-tree-file',   type=str, help='prefix tree file',              default='model/prefix-tree')
    args = parser.parse_args()
    input_folder = args.input_folder
    message_size = args.message_size
    batch_size = args.batch_size
    command = args.command
    model_file = args.model_file
    model_file = model_file if not model_file is None else '/tmp/model/model' if command == 'train' else 'model/model-1/model'
    prefix_tree_file = args.prefix_tree_file

    if command == 'train':
        train_network()
    elif command == 'test':
        test_network()
    elif command == 'play':
        play(prefix_tree_file)
    else:
        raise ValueError(command)
