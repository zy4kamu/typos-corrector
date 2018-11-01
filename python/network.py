import argparse, os
import tensorflow as tf
import numpy as np
import utils
import cpp_bindings

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops

ARGS = None

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
        self._left_matrix = self.add_variable("left_matrix", shape=[input_depth + h_depth, ARGS.compressor_size])
        self._right_matrix = self.add_variable("right_matrix", shape=[ARGS.compressor_size, 4 * self._num_units])
        self._kernel = math_ops.mat_mul(self._left_matrix, self._right_matrix)
        self._bias = self.add_variable("bias",shape=[4 * self._num_units],
                                       initializer=init_ops.zeros_initializer(dtype=self.dtype))

class Network(object):
    def __init__(self):
        self.parametes = {}
        self.sess = tf.Session()

    def train(self, restore_parameters_from_file):
        self.__initialize_for_training()
        train_num_correct = 0
        train_num_letters = 0

        clean_test_batch, contaminated_test_batch = cpp_bindings.generate_random_batch(ARGS.test_batch_size)

        with tf.device("/device:GPU:0"):
            test_logits = self.__create_output_logits(ARGS.test_batch_size)
            train_logits = self.__create_output_logits(ARGS.batch_size)

            # create loss which we want to minimize and optimizer to make optimization
            clean_one_hot_embedding = tf.one_hot(self.clean_tokens, utils.NUM_SYMBOLS)
            total_loss = None
            for i in range(ARGS.message_size):
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_logits[i], labels=clean_one_hot_embedding[:, i, :])
                loss = tf.reduce_mean(loss)
                if total_loss is None: total_loss = loss
                else: total_loss += loss
            total_loss /= ARGS.message_size
            optimizer = tf.train.AdamOptimizer().minimize(total_loss)

        self.__initialize_parameters_for_save()

        saver = tf.train.Saver()
        model_folder = 'models/tensorflow-binaries/{}'.format(ARGS.country)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_file = os.path.join(model_folder, 'model')
        if restore_parameters_from_file:
            saver.restore(self.sess, model_file)
            counter = ARGS.test_num_iterations
        else:
            initializer = tf.global_variables_initializer()
            self.sess.run(initializer)
            counter = 0
        max_accuracy = 0
        test_accuracy = 0
        while True:
            # make intermediate report
            if counter == ARGS.test_num_iterations:
                print '\r',

                # print current model
                should_save_model = test_accuracy > max_accuracy
                if should_save_model:
                    max_accuracy = test_accuracy
                    saver.save(self.sess, model_file)
                    self.save_parameters_to_file()

                # print statistics on train
                print '\ntrain: {} correct of {}; accuracy = {}'.format(train_num_correct, train_num_letters,
                                                                        float(train_num_correct) / float(train_num_letters + 1))
                counter = 0
                train_num_correct = 0
                train_num_letters = 0

                # print statistics on test
                test_num_correct = 0
                test_num_letters = 0
                dummy_num_correct = 0
                predictions, = self.sess.run([test_logits], feed_dict={ self.contaminated_tokens:contaminated_test_batch,
                                                                        self.clean_tokens:clean_test_batch })
                test_num_letters += ARGS.test_batch_size * ARGS.message_size
                for i in range(ARGS.message_size):
                    test_num_correct += np.sum((np.argmax(predictions[i], axis=1) == clean_test_batch[:, i]))
                    dummy_num_correct += np.sum(clean_test_batch[:, i] == contaminated_test_batch[:, i])

                # create all_predicted
                all_predicted = np.empty(dtype=np.int32, shape=(ARGS.test_batch_size, ARGS.message_size))
                for i in range(ARGS.message_size):
                    all_predicted[:, i] = np.argmax(predictions[i], axis=1)

                # create calculating levenstein
                sum_levenstein = 0
                for i in range(ARGS.test_batch_size):
                    predicted_token = utils.numpy_to_string(all_predicted[i, :])
                    true_token = utils.numpy_to_string(clean_test_batch[i, :])
                    sum_levenstein += cpp_bindings.levenstein(predicted_token, true_token)

                # create first mistake statistics file
                if should_save_model:
                    first_mistake_statistics = [0] * (ARGS.message_size + 1)
                    for i in range(ARGS.test_batch_size):
                        predicted_token = utils.numpy_to_string(all_predicted[i, :])
                        true_token = utils.numpy_to_string(clean_test_batch[i, :])
                        for i in range(ARGS.message_size):
                            if predicted_token[i] != true_token[i]:
                                first_mistake_statistics[i] += 1
                                break
                            if i + 1 == ARGS.message_size:
                                first_mistake_statistics[-1] += 1
                        with open('models/binaries/{}/first-mistake-statistics'.format(ARGS.country), 'w') as writer:
                            writer.write('\n'.join([str(_) for _ in first_mistake_statistics]))

                test_accuracy = 100.0 * float(test_num_correct) / float(test_num_letters)
                print 'test: {} correct of {}; accuracy = {}%; max accuracy = {}'.format(test_num_correct, test_num_letters,
                                                                                         test_accuracy, max_accuracy)
                print 'dummy: {} correct of {}; accuracy = {}'.format(dummy_num_correct, test_num_letters,
                                                                      float(dummy_num_correct) / float(test_num_letters))
                print 'levenstein: {}'.format(float(sum_levenstein) / float(ARGS.test_batch_size))
                print ''

                if test_accuracy > ARGS.stop_accuracy:
                    print 'Reached target accuracy {}; exit'.format(ARGS.stop_accuracy)
                    exit(0)

            # update gradient
            clean, contaminated = cpp_bindings.generate_random_batch(ARGS.batch_size)
            _, predictions, l = self.sess.run([optimizer, train_logits, total_loss],
                feed_dict={ self.clean_tokens:clean, self.contaminated_tokens:contaminated })

            # update statistics on train
            counter += 1
            print '\r', counter,
            train_num_letters += ARGS.batch_size * ARGS.message_size
            for i in range(ARGS.message_size):
                train_num_correct += np.sum((np.argmax(predictions[i], axis=1) == clean[:, i]))

    def __initialize_parameters_for_save(self):
        def add_parameter(tensor, tensor_name):
            tf.identity(tensor, tensor_name)
            self.parametes[tensor_name] = tensor
        add_parameter(self.encode_lstm._left_matrix,  'encode_lstm_left_matrix')
        add_parameter(self.encode_lstm._right_matrix, 'encode_lstm_right_matrix')
        add_parameter(self.encode_lstm._bias,         'encode_lstm_bias')
        add_parameter(self.decode_lstm._left_matrix,  'decode_lstm_left_matrix')
        add_parameter(self.decode_lstm._right_matrix, 'decode_lstm_right_matrix')
        add_parameter(self.decode_lstm._bias,         'decode_lstm_bias')
        add_parameter(self.hidden_layer_weights,      'hidden_layer_weights')
        add_parameter(self.hidden_layer_bias,         'hidden_layer_bias')

    def save_parameters_to_file(self):
        output_folder = 'models/binaries/' + ARGS.country
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for k, v in self.parametes.iteritems():
            [parameter] = self.sess.run([v], feed_dict={})
            parameter.tofile(os.path.join(output_folder, k))

    def __initialize_for_training(self):
        with tf.device("/device:GPU:0"):
            self.clean_tokens = tf.placeholder(tf.int32, [None, ARGS.message_size], name='clean_tokens')
            self.contaminated_tokens = tf.placeholder(tf.int32, [None, ARGS.message_size], name='contaminated_tokens')

            self.contaminated_embedding = tf.one_hot(self.contaminated_tokens, utils.NUM_SYMBOLS)

            self.encode_lstm = CompressedLSTM(ARGS.lstm_size)
            self.decode_lstm = CompressedLSTM(ARGS.lstm_size)
            self.hidden_layer_weights = tf.Variable(tf.truncated_normal([ARGS.lstm_size, utils.NUM_SYMBOLS]))
            self.hidden_layer_bias  = tf.Variable(tf.truncated_normal([utils.NUM_SYMBOLS]))

    def __create_output_logits(self, batch_size):
        clean_embedding = tf.one_hot(self.clean_tokens, utils.NUM_SYMBOLS)
        logits = []
        state = self.encode_lstm.zero_state(batch_size, tf.float32)
        for i in range(ARGS.message_size):
            output, state = self.encode_lstm(self.contaminated_embedding[:, ARGS.message_size - i - 1, :], state)
        for i in range(ARGS.message_size):
            logits.append(tf.matmul(output, self.hidden_layer_weights) + self.hidden_layer_bias)
            output, state = self.decode_lstm(clean_embedding[:, i, :], state)
        return logits


def generate_cpp_bindings():
    cpp_bindings.generate_cpp_bindings(country=ARGS.country,
                                       mistake_probability=ARGS.mistake_probability,
                                       message_size=ARGS.message_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural network for typos correction')
    parser.add_argument('-c', '--command',             type=str,   help='command to process',            required=True,
                        choices=['train', 'continue'])
    parser.add_argument('-o', '--country',             type=str,   help='country to process',            default='united states of america/illinois'),
    parser.add_argument('-m', '--message-size',        type=int,   help='length of each token in batch', default=25)
    parser.add_argument('-b', '--batch-size',          type=int,   help='number of tokens in batch',     default=1024)
    parser.add_argument('-p', '--mistake-probability', type=float, help='mistake probability',           default=0.2)
    parser.add_argument('-s', '--compressor-size',     type=int,   help='size of internal LSTM state',   default=128)
    parser.add_argument('-l', '--lstm-size',           type=int,   help='LSTM cell size',                default=512)
    parser.add_argument('-t', '--test-batch-size',     type=int,   help='test batch size',               default=10000)
    parser.add_argument('-n', '--test-num-iterations', type=int,   help='test number of iterations',     default=2500)
    parser.add_argument('-a', '--stop-accuracy',       type=float, help='stop when reach this accuracy', default=99.97)
    ARGS = parser.parse_args()

    generate_cpp_bindings()

    if ARGS.command == 'train':
        network = Network()
        network.train(restore_parameters_from_file=False)
    elif ARGS.command == 'continue':
        network = Network()
        network.train(restore_parameters_from_file=True)

