import argparse, os
import tensorflow as tf
import numpy as np
import utils
from dataset_generator import DataSetGenerator

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops

command             = None
input_folder        = None
message_size        = 30
batch_size          = None
model_file          = 'model/model-1/model'
test_num_iterations = 2500
test_batch_size     = 10000
lstm_size           = 1024

#TODO: 1. remove char_embedding: one hot is enough
#TODO: 2. insert handcrafted LSTM and measure quality
#TODO: 3. reduce rank of all matrices

class CompressedLSTM(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None, name=None):
        super(CompressedLSTM, self).__init__(num_units, forget_bias, state_is_tuple, activation, reuse, name)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)
        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._left_matrix = self.add_variable("left_matrix", shape=[input_depth + h_depth, 64])
        self._right_matrix = self.add_variable("right_matrix", shape=[64, 4 * self._num_units])
        self._kernel = math_ops.mat_mul(self._left_matrix, self._right_matrix)
        self._bias = self.add_variable("bias",shape=[4 * self._num_units],
                                       initializer=init_ops.zeros_initializer(dtype=self.dtype))

# Not used yet
class LSTM(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Input gate: input, previous output, and bias.
        self.ix = tf.Variable(tf.truncated_normal([input_size, output_size], -0.1, 0.1))
        self.im = tf.Variable(tf.truncated_normal([output_size, output_size], -0.1, 0.1))
        self.ib = tf.Variable(tf.zeros([1, output_size]))
        # Forget gate: input, previous output, and bias.
        self.fx = tf.Variable(tf.truncated_normal([input_size, output_size], -0.1, 0.1))
        self.fm = tf.Variable(tf.truncated_normal([output_size, output_size], -0.1, 0.1))
        self.fb = tf.Variable(tf.zeros([1, output_size]))
        # Memory cell: input, state and bias.
        self.cx = tf.Variable(tf.truncated_normal([input_size, output_size], -0.1, 0.1))
        self.cm = tf.Variable(tf.truncated_normal([output_size, output_size], -0.1, 0.1))
        self.cb = tf.Variable(tf.zeros([1, output_size]))
        # Output gate: input, previous output, and bias.
        self.ox = tf.Variable(tf.truncated_normal([input_size, output_size], -0.1, 0.1))
        self.om = tf.Variable(tf.truncated_normal([output_size, output_size], -0.1, 0.1))
        self.ob = tf.Variable(tf.zeros([1, output_size]))

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.state = tf.zeros([batch_size, self.output_size])
        self.output = tf.zeros([batch_size, self.output_size])

    def lstm_cell(self, i):
        forget_gate = tf.sigmoid(tf.matmul(i, self.fx) + tf.matmul(self.output, self.fm) + self.fb)
        self.state = forget_gate * self.state

        input_gate = tf.sigmoid(tf.matmul(i, self.ix) + tf.matmul(self.output, self.im) + self.ib)
        update = tf.matmul(i, self.cx) + tf.matmul(self.output, self.cm) + self.cb
        self.state = self.state + input_gate * tf.tanh(update)

        output_gate = tf.sigmoid(tf.matmul(i, self.ox) + tf.matmul(self.output, self.om) + self.ob)
        self.output = output_gate * tf.tanh(self.state)
        return self.output

class Network(object):
    def __init__(self):
        self.sess = tf.Session()

    def train(self):
        self.__initialize_for_training()
        self.__initialize_tensors_for_online_application()
        batch_generator = DataSetGenerator(dictionary_file='model/dictionary', message_size=message_size, mistake_probability=0.2)
        counter = 0
        train_num_correct = 0
        train_num_letters = 0

        clean_test_batch, contaminated_test_batch = batch_generator.next(test_batch_size)

        with tf.device("/device:GPU:0"):
            test_logits = self.__create_output_logits(test_batch_size)
            train_logits = self.__create_output_logits(batch_size)

            # create loss which we want to minimize and optimizer to make optimization
            clean_one_hot_embedding = tf.one_hot(self.clean_tokens, utils.NUM_SYMBOLS)
            total_loss = None
            for i in range(message_size):
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_logits[i], labels=clean_one_hot_embedding[:, i, :])
                loss = tf.reduce_mean(loss)
                if total_loss is None: total_loss = loss
                else: total_loss += loss
            total_loss /= message_size
            optimizer = tf.train.AdamOptimizer().minimize(total_loss)

        initializer = tf.global_variables_initializer()
        self.sess.run(initializer)
        while True:
            # update gradient
            clean, contaminated = batch_generator.next(batch_size)
            _, predictions, l = self.sess.run([optimizer, train_logits, total_loss],
                feed_dict={ self.clean_tokens:clean, self.contaminated_tokens:contaminated })

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
                saver.save(self.sess, model_file)

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
                predictions, = self.sess.run([test_logits], feed_dict={ self.contaminated_tokens:contaminated_test_batch,
                                                                        self.clean_tokens:clean_test_batch })
                test_num_letters += test_batch_size * message_size
                for i in range(message_size):
                    test_num_correct += np.sum((np.argmax(predictions[i], axis=1) == clean_test_batch[:, i]))
                    dummy_num_correct += np.sum(clean_test_batch[:, i] == contaminated_test_batch[:, i])

                sum_levenstein = 0
                all_predicted = np.empty(dtype=np.int32, shape=(test_batch_size, message_size))
                for i in range(message_size):
                    all_predicted[:, i] = np.argmax(predictions[i], axis=1)
                for i in range(test_batch_size):
                    predicted_token = utils.numpy_to_string(all_predicted[i, :])
                    true_token = utils.numpy_to_string(clean_test_batch[i, :])
                    sum_levenstein += utils.levenstein(predicted_token, true_token)

                print 'test: {} correct of {}; accuracy = {}'.format(test_num_correct, test_num_letters,
                                                                     float(test_num_correct) / float(test_num_letters))
                print 'dummy: {} correct of {}; accuracy = {}'.format(dummy_num_correct, test_num_letters,
                                                                      float(dummy_num_correct) / float(test_num_letters))
                print 'levenstein: {}'.format(float(sum_levenstein) / float(test_batch_size))
                print ''

    def test(self):
        network.read_all_required_tensors_from_file()
        batch_generator = DataSetGenerator(dictionary_file='model/dictionary', message_size=message_size, mistake_probability=0.2)
        clean_test_batch, contaminated_test_batch = batch_generator.next(test_batch_size)
        dummy_levenstein_sum = 0
        predicted_levenstein_sum = 0
        for i in range(test_batch_size):
            print '\r', i, 'of', test_batch_size,
            clean_token = utils.numpy_to_string(clean_test_batch[i, :])
            contaminated_token = utils.numpy_to_string(contaminated_test_batch[i, :])
            predicted_token = self.make_prediction_for_token(clean_token)
            dummy_levenstein_sum += utils.levenstein(clean_token, contaminated_token)
            predicted_levenstein_sum += utils.levenstein(clean_token, predicted_token)
        print 'dummy levenstein: {}'.format(float(dummy_levenstein_sum) / float(test_batch_size))
        print 'predicted levenstein: {}'.format(float(predicted_levenstein_sum) / float(test_batch_size))

    def read_all_required_tensors_from_file(self):
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

    def make_prediction_for_token(self, token, verbose=False, allow_mistake=0.01):
        # convert token to numpy array
        token += ' ' * (message_size - len(token))
        numpy_token = np.ones(message_size, dtype=np.int32) * utils.SPACE_INT
        numpy_token[0:len(token)] = utils.string_to_numpy(token)
        numpy_token = numpy_token.reshape((-1, message_size))

        # create intial state (encode)
        initial_lstm_c, initial_lstm_h, initial_logits = self.sess.run(
            [self.initial_play_state_c, self.initial_play_state_h, self.initial_logits],
            feed_dict={self.contaminated_tokens:numpy_token})

        # pass over token and decode token (decode)
        best_hypo = ''
        probs = []
        lstm_c, lstm_h, logits = initial_lstm_c, initial_lstm_h, initial_logits
        for t in range(message_size):
            best_char_index = np.argmax(logits[0, :])
            probs.append(np.exp(logits[0,:]) / np.sum(np.exp(logits[0, :])))
            numpy_char = np.ones(dtype=np.int32, shape=(1,)) * best_char_index
            lstm_c, lstm_h, logits = self.sess.run(
                [self.apply_output_state_c, self.apply_output_state_h, self.apply_output_logits],
                feed_dict={self.apply_input_state_c:lstm_c, self.apply_input_state_h:lstm_h, self.apply_input_char:numpy_char})
            best_hypo += utils.int_to_char(best_char_index)

        # trick to allow 1 mistake in word
        other_hypos = []
        if not allow_mistake is None:
            allowed_mistakes = []
            for t in range(message_size):
                for ch in range(utils.NUM_SYMBOLS):
                    max_probs = np.max(probs[t])
                    if probs[t][ch] > allow_mistake and probs[t][ch] + 1e-5 < max_probs:
                        allowed_mistakes.append((t, ch, probs[t][ch]))
            allowed_mistakes = sorted(allowed_mistakes, key=lambda (t, ch, p): -p)
            for t, ch, p in allowed_mistakes:
                mistakened_prefix = ''
                lstm_c, lstm_h, logits = initial_lstm_c, initial_lstm_h, initial_logits
                for tt in range(message_size):
                    best_char_index = ch if t == tt else np.argmax(logits[0, :])
                    probs.append(np.exp(logits[0,:]) / np.sum(np.exp(logits[0, :])))
                    numpy_char = np.ones(dtype=np.int32, shape=(1,)) * best_char_index
                    lstm_c, lstm_h, logits = self.sess.run(
                        [self.apply_output_state_c, self.apply_output_state_h, self.apply_output_logits],
                        feed_dict={self.apply_input_state_c:lstm_c, self.apply_input_state_h:lstm_h, self.apply_input_char:numpy_char})
                    mistakened_prefix += utils.int_to_char(best_char_index)
                if utils.levenstein(mistakened_prefix, token) < 4:
                    other_hypos.append(mistakened_prefix)

        # print table of probabilities
        if verbose:
            def format(number):
                return '....' if number < 0.03 else '{:1.2f}'.format(number)
            print ' ', ':', '  '.join(['{:4.0f}'.format(_) for _ in range(message_size)])
            for ch in range(utils.NUM_SYMBOLS):
                print utils.int_to_char(ch), ':', '  '.join([format(probs[t][ch]) for t in range(message_size)])
        return best_hypo, other_hypos

    def __create_output_logits(self, batch_size):
        clean_embedding = tf.nn.embedding_lookup(self.char_embedding_matrix, self.clean_tokens)
        logits = []
        state = self.encode_lstm.zero_state(batch_size, tf.float32)
        for i in range(message_size):
            output, state = self.encode_lstm(self.contaminated_embedding[:, message_size - i - 1, :], state)
        for i in range(message_size):
            logits.append(tf.matmul(output, self.hidden_layer_weights) + self.hidden_layer_bias)
            output, state = self.decode_lstm(clean_embedding[:, i, :], state)
        return logits

    def __initialize_for_training(self):
        with tf.device("/device:GPU:0"):
            self.clean_tokens = tf.placeholder(tf.int32, [None, message_size], name='clean_tokens')
            self.contaminated_tokens = tf.placeholder(tf.int32, [None, message_size], name='contaminated_tokens')

            self.char_embedding_matrix = tf.Variable(tf.random_uniform([utils.NUM_SYMBOLS, utils.NUM_SYMBOLS], -1.0, 1.0))
            self.contaminated_embedding = tf.nn.embedding_lookup(self.char_embedding_matrix, self.contaminated_tokens)

            self.encode_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.decode_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.hidden_layer_weights = tf.Variable(tf.truncated_normal([lstm_size, utils.NUM_SYMBOLS]))
            self.hidden_layer_bias  = tf.Variable(tf.truncated_normal([utils.NUM_SYMBOLS]))

    def __initialize_tensors_for_online_application(self):
        # initial play state
        initial_play_state = self.encode_lstm.zero_state(1, tf.float32)
        for i in range(message_size):
            output, initial_play_state = self.encode_lstm(self.contaminated_embedding[:, message_size - i - 1, :], initial_play_state)
        tf.identity(initial_play_state.c, 'initial_play_state_c')
        tf.identity(initial_play_state.h, 'initial_play_state_h')
        tf.identity(tf.matmul(output, self.hidden_layer_weights) + self.hidden_layer_bias, 'initial_logits')

        # apply lstm cell
        apply_input_state_c = tf.placeholder(dtype=tf.float32, shape=(1, lstm_size), name='apply_input_state_c')
        apply_input_state_h = tf.placeholder(dtype=tf.float32, shape=(1, lstm_size), name='apply_input_state_h')
        apply_input_char = tf.placeholder(dtype=np.int32, shape=(1,), name='apply_input_char')
        apply_input_state = tf.contrib.rnn.LSTMStateTuple(apply_input_state_c, apply_input_state_h)
        apply_input_embedding = tf.nn.embedding_lookup(self.char_embedding_matrix, apply_input_char)
        apply_output_logits, apply_output_state = self.decode_lstm(apply_input_embedding, apply_input_state)
        tf.identity(tf.matmul(apply_output_logits, self.hidden_layer_weights) + self.hidden_layer_bias, 'after_apply_logits')
        tf.identity(apply_output_state.c, 'after_apply_state_c')
        tf.identity(apply_output_state.h, 'after_apply_state_h')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test neural network for typos correction')
    parser.add_argument('-c', '--command',          type=str, help='command to process',            required=True, choices=['train', 'play', 'test'])
    parser.add_argument('-i', '--input-folder',     type=str, help='folder with binary batches',    default='../dataset')
    parser.add_argument('-m', '--message-size',     type=int, help='length of each token in batch', default=30)
    parser.add_argument('-b', '--batch-size',       type=int, help='number of tokens in batch',     default=128)
    parser.add_argument('-f', '--model-file',       type=str, help='file with binary model',        default=None)
    args = parser.parse_args()
    input_folder = args.input_folder
    message_size = args.message_size
    batch_size = args.batch_size
    command = args.command
    model_file = args.model_file
    model_file = model_file if not model_file is None else 'model/model-1/model'

    network = Network()

    if command == 'train':
        network.train()
    elif command == 'play':
        network.read_all_required_tensors_from_file()
        while True:
            token = raw_input("Input something: ")
            best_hypo, other_hypos = network.make_prediction_for_token(token)
            print best_hypo
            if len(other_hypos) > 0:
                print ''
                print 'other hypos:'
                print '\n'.join(other_hypos)
    elif command == 'test':
        network.test()
    else:
        raise ValueError(command)
