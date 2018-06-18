import argparse, os
import time
import tensorflow as tf
import numpy as np
from convertions import NUM_SYMBOLS, string_to_numpy, SPACE_INT, numpy_to_string, int_to_char, char_to_int
from dataset_generator import DataSetGenerator
from prefix_tree import PrefixTree, get_prefix_tree_root, get_transitions, make_transition

command      = None
input_folder = None
message_size = None
batch_size   = None
model_file   = None
prefix_tree  = None
prefix_tree_automata = None
num_hypos    = 1000
test_num_iterations = 2500
test_batch_size = 10000

#TODO: - 1. deal with mistakes on the first characters
#TODO: - 2. memory map prefix tree to file
#TODO: + 3. LSTM
#TODO: + 4. testing with pagerank
#TODO: + 5. doesn't work with absent spaces yet
#TODO: | 6. learning by prefixes
#TODO: - 7. port everything to pure C++
#TODO: + 8. print mistakened tokens
#TODO: - 9. creation of prefix tree is not ported from preifx-tree project
#TODO: + 10. sort hypothesis by leventstein and only then by score
#TODO: - 11. BUG: convert logits to log probs
#TODO: - 12. Prefix bug in prefix tree

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
        clean_embedding = tf.nn.embedding_lookup(char_embedding_matrix, clean_tokens)

        # 97.5% with real token input
        lstm_size = 512
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        hidden1 = tf.Variable(tf.truncated_normal([lstm_size, NUM_SYMBOLS]))
        biases1  = tf.Variable(tf.truncated_normal([NUM_SYMBOLS]))
        def create_architecture(batch_size):
            logits = []
            state = lstm.zero_state(batch_size, tf.float32)
            for i in range(message_size):
                output, state = lstm(contaminated_embedding[:, message_size - i - 1, :], state)
            for i in range(message_size):
                logits.append(tf.matmul(output, hidden1) + biases1)
                output, state = lstm(clean_embedding[:, i, :], state)
            return logits

        # training
        train_logits = create_architecture(batch_size)
        total_loss = None
        for i in range(message_size):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_logits[i], labels=clean_one_hot_embedding[:, i, :])
            loss = tf.reduce_sum(loss)
            if total_loss is None: total_loss = loss
            else: total_loss += loss
        optimizer = tf.train.AdamOptimizer().minimize(total_loss)

        # testing
        test_logits = create_architecture(test_batch_size)

        # initial play state
        initial_play_state = lstm.zero_state(1, tf.float32)
        for i in range(message_size):
            output, initial_play_state = lstm(contaminated_embedding[:, message_size - i - 1, :], initial_play_state)
        initial_play_state_c = tf.identity(initial_play_state.c, 'initial_play_state_c')
        initial_play_state_h = tf.identity(initial_play_state.h, 'initial_play_state_h')
        initial_output = tf.identity(tf.matmul(output, hidden1) + biases1, 'initial_logits')

        # apply lstm cell
        apply_input_state_c = tf.placeholder(dtype=tf.float32, shape=(1, lstm_size), name='apply_input_state_c')
        apply_input_state_h = tf.placeholder(dtype=tf.float32, shape=(1, lstm_size), name='apply_input_state_h')
        apply_input_char = tf.placeholder(dtype=np.int32, shape=(1,), name='apply_input_char')
        apply_input_state = tf.contrib.rnn.LSTMStateTuple(apply_input_state_c, apply_input_state_h)
        apply_input_embedding = tf.nn.embedding_lookup(char_embedding_matrix, apply_input_char)
        apply_output_logits, apply_output_state = lstm(apply_input_embedding, apply_input_state)
        apply_output_logits = tf.identity(tf.matmul(apply_output_logits, hidden1) + biases1, 'after_apply_logits')
        apply_output_state_c = tf.identity(apply_output_state.c, 'after_apply_state_c')
        apply_output_state_h = tf.identity(apply_output_state.h, 'after_apply_state_h')

    batch_generator = DataSetGenerator(dictionary_file='model/dictionary', message_size=message_size, mistake_probability=0.2)
    clean_test_batch, contaminated_test_batch = batch_generator.next(test_batch_size)
    counter = 0
    train_num_correct = 0
    train_num_letters = 0

    # train network
    initializer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initializer)
        while True:
            # update gradient
            clean, contaminated = batch_generator.next(batch_size)
            _, predictions, l = sess.run([optimizer, train_logits, total_loss],
                    feed_dict={clean_tokens:clean, contaminated_tokens:contaminated})

            # update statistics on train
            counter += 1
            print '\r', counter,
            train_num_letters += batch_size * message_size
            for i in range(message_size):
                train_num_correct += np.sum((np.argmax(predictions[i], axis=1) == clean[:, i]))

            # make intermediate report
            if counter == test_num_iterations:
                # print current model
                print '\r',
                saver = tf.train.Saver()
                saver.save(sess, model_file)

                # print statistics on train
                print '\ntrain: {} correct of {}; accuracy = {}'.format(train_num_correct, train_num_letters,
                                                                        float(train_num_correct) / float(train_num_letters))
                counter = 0
                train_num_correct = 0
                train_num_letters = 0

                # print statistics on test
                test_num_correct = 0
                test_num_letters = 0
                dummy_num_correct = 0
                predictions, = sess.run([test_logits], feed_dict={ contaminated_tokens:contaminated_test_batch,
                                                                   clean_tokens:clean_test_batch })
                test_num_letters += test_batch_size * message_size
                for i in range(message_size):
                    test_num_correct += np.sum((np.argmax(predictions[i], axis=1) == clean_test_batch[:, i]))
                    dummy_num_correct += np.sum(clean_test_batch[:, i] == contaminated_test_batch[:, i])
                print 'test: {} correct of {}; accuracy = {}'.format(test_num_correct, test_num_letters,
                                                                     float(test_num_correct) / float(test_num_letters))
                print 'dummy: {} correct of {}; accuracy = {}'.format(dummy_num_correct, test_num_letters,
                                                                      float(dummy_num_correct) / float(test_num_letters))

                """
                # full tests
                with open('/tmp/mistakened_tokens', 'w') as writer:
                    num_found_tokens = 0
                    for token_index in range(test_batch_size):
                        np_predictions = np.empty(dtype=np.double, shape=[len(predictions), NUM_SYMBOLS])
                        for letter_index in range(message_size):
                            np_predictions[letter_index, :] = predictions[letter_index][token_index, :]
                        clean_token = numpy_to_string(clean_test_batch[token_index, :]).strip()
                        viterbi_tokens, viterbi_logits = prefix_tree.viterbi(np_predictions, num_hypos)
                        if clean_token in viterbi_tokens:
                            num_found_tokens += 1
                        else:
                            contaminated_token = numpy_to_string(contaminated_test_batch[token_index, :]).strip()
                            writer.write('{} {}\n'.format(clean_token, contaminated_token))
                print 'full: {} correct of {}; accuracy = {}'.format(num_found_tokens, test_batch_size,
                                                                     float(num_found_tokens) / float(test_batch_size))
                """
                print ''

class AutomataState(object):
    def __init__(self, lstm_c, lstm_h, logits, full_prefix_logits=0, prefix=''):
        self.lstm_c = lstm_c
        self.lstm_h = lstm_h
        self.logits = logits
        #self.prefix_state = prefix_state
        self.full_log_probability = full_prefix_logits
        self.prefix = prefix

    #def get_transitions(self):
    #    return get_transitions(self.prefix_state)

class AutomataSession(object):
    def __init__(self):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(model_file + '.meta', clear_devices=True)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(model_file + '.meta')))
        self.graph = tf.get_default_graph()
        self.contaminated_tokens = self.graph.get_tensor_by_name("contaminated_tokens:0")
        self.initial_play_state_c = self.graph.get_tensor_by_name('initial_play_state_c:0')
        self.initial_play_state_h = self.graph.get_tensor_by_name('initial_play_state_h:0')
        self.initial_logits = self.graph.get_tensor_by_name('initial_logits:0')
        self.apply_output_logits = self.graph.get_tensor_by_name('after_apply_logits:0')
        self.apply_output_state_c = self.graph.get_tensor_by_name('after_apply_state_c:0')
        self.apply_output_state_h = self.graph.get_tensor_by_name('after_apply_state_h:0')
        self.apply_input_state_c = self.graph.get_tensor_by_name('apply_input_state_c:0')
        self.apply_input_state_h = self.graph.get_tensor_by_name('apply_input_state_h:0')
        self.apply_input_char = self.graph.get_tensor_by_name('apply_input_char:0')

    def feed_token(self, token):
        numpy_token = np.ones(message_size, dtype=np.int32) * SPACE_INT
        numpy_token[0:len(token)] = string_to_numpy(token)
        numpy_token = numpy_token.reshape((-1, message_size))
        lstm_c, lstm_h, first_char_logits = self.sess.run(
            [self.initial_play_state_c, self.initial_play_state_h, self.initial_logits],
            feed_dict={self.contaminated_tokens:numpy_token})
        return AutomataState(lstm_c, lstm_h, first_char_logits, get_prefix_tree_root())

    def apply(self, state, ch):
        char = np.empty(dtype=np.int32, shape=(1,))
        char[0] = char_to_int(ch)
        lstm_c, lstm_h, next_char_logits = self.sess.run(
            [self.apply_output_state_c, self.apply_output_state_h, self.apply_output_logits],
            feed_dict={self.apply_input_state_c:state.lstm_c, self.apply_input_state_h:state.lstm_h, self.apply_input_char:char})
        return AutomataState(lstm_c, lstm_h, next_char_logits,
                             state.full_log_probability + state.logits[0, char[0]], state.prefix + ch)

    def find_best_hypos(self, token, num_hypos=100):
        token += ' ' * (message_size - len(token))
        zz = self.feed_token(token)
        for i in range(message_size):
            ch = int_to_char(np.argmax(zz.logits[0, :]))
            zz = self.apply(zz, ch)
        print zz.prefix

def play():
    sess = AutomataSession()
    while True:
        token = raw_input("Input something: ")
        sess.find_best_hypos(token)

        """
        # feed the same token
        best_predicted = np.empty(dtype=np.int32, shape=(message_size,))
        state = sess.feed_token(token)
        for i in range(message_size):
            best_predicted[i] = np.argmax(state.logits)
            state = sess.apply(state, token[i])
        print numpy_to_string(best_predicted), state.full_log_probability

        # feed slightly contaminated
        best_predicted = np.empty(dtype=np.int32, shape=(message_size,))
        state = sess.feed_token(token)
        contaminated_token = ('z' + token)[0:message_size]
        for i in range(message_size):
            best_predicted[i] = np.argmax(state.logits)
            state = sess.apply(state, contaminated_token[i])
        print numpy_to_string(best_predicted), state.full_log_probability

        # feed random characters
        best_predicted = np.empty(dtype=np.int32, shape=(message_size,))
        state = sess.feed_token(token)
        for i in range(message_size):
            best_predicted[i] = np.argmax(state.logits)
            state = sess.apply(state, 'a')
        print numpy_to_string(best_predicted), state.full_log_probability
        """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test neural network for typos correction')
    parser.add_argument('-c', '--command',          type=str, help='command to process',            required=True, choices=['train', 'play'])
    parser.add_argument('-i', '--input-folder',     type=str, help='folder with binary batches',    default='../dataset')
    parser.add_argument('-m', '--message-size',     type=int, help='length of each token in batch', default=30)
    parser.add_argument('-b', '--batch-size',       type=int, help='number of tokens in batch',     default=128)
    parser.add_argument('-f', '--model-file',       type=str, help='file with binary model',        default=None)
    parser.add_argument('-t', '--prefix-tree-file', type=str, help='prefix tree file',              default='model/prefix-tree')
    args = parser.parse_args()
    input_folder = args.input_folder
    message_size = args.message_size
    batch_size = args.batch_size
    command = args.command
    model_file = args.model_file
    model_file = model_file if not model_file is None else 'model/model-1/model'
    prefix_tree = PrefixTree(args.prefix_tree_file)

    if command == 'train':
        train_network()
    elif command == 'play':
        play()
    else:
        raise ValueError(command)
