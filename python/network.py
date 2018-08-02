import argparse, os
import tensorflow as tf
import numpy as np
import utils
import cpp_bindings
import protobuf_talker

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops

message_size = 25
batch_size = None
model_file = 'model/model-1/model'
test_num_iterations = 2500
test_batch_size = 10000
lstm_size = 512


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


class Network(object):
    def __init__(self):
        self.sess = tf.Session()

    def train(self):
        self.__initialize_for_training()
        self.__initialize_tensors_for_online_application()
        counter = 0
        train_num_correct = 0
        train_num_letters = 0

        clean_test_batch, contaminated_test_batch = cpp_bindings.generate_random_batch(test_batch_size,
                                                                                       use_one_update_region=False)

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
            clean, contaminated = cpp_bindings.generate_random_batch(batch_size,
                                                                     use_one_update_region=False)
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
                    sum_levenstein += cpp_bindings.levenstein(predicted_token, true_token)

                print 'test: {} correct of {}; accuracy = {}'.format(test_num_correct, test_num_letters,
                                                                     float(test_num_correct) / float(test_num_letters))
                print 'dummy: {} correct of {}; accuracy = {}'.format(dummy_num_correct, test_num_letters,
                                                                      float(dummy_num_correct) / float(test_num_letters))
                print 'levenstein: {}'.format(float(sum_levenstein) / float(test_batch_size))
                print ''

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

class NetworkAutomata(Network):
    def __init__(self):
        Network.__init__(self)
        self._default_mistake_counter = DefaultMistakeCounter()

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

        self._reset()

    def encode(self, token):
        self._reset()
        # convert token to numpy array
        token += ' ' * (message_size - len(token))
        numpy_token = np.ones(message_size, dtype=np.int32) * utils.SPACE_INT
        numpy_token[0:len(token)] = utils.string_to_numpy(token)
        numpy_token = numpy_token.reshape((-1, message_size))

        # create intial state (encode)
        self._lstm_c, self._lstm_h, logits = self.sess.run(
            [self.initial_play_state_c, self.initial_play_state_h, self.initial_logits],
            feed_dict={self.contaminated_tokens:numpy_token})

        self._probs = [np.exp(logits[0,:]) / np.sum(np.exp(logits[0, :]))]
        self._break_logits.append(np.log(self._probs[-1]) + self._default_mistake_counter.get(1))
        return self._probs[-1]

    def apply(self, letter):
        self._input_history += letter
        numpy_char = np.ones(dtype=np.int32, shape=(1,)) * utils.char_to_int(letter)
        self._lstm_c, self._lstm_h, logits = self.sess.run(
            [self.apply_output_state_c, self.apply_output_state_h, self.apply_output_logits],
            feed_dict={self.apply_input_state_c:self._lstm_c, self.apply_input_state_h:self._lstm_h, self.apply_input_char:numpy_char})
        length = len(self._input_history)
        new_probs = np.exp(logits[0,:]) / np.sum(np.exp(logits[0, :]))
        if length < message_size:
            self._break_logits.append(self._history_logit + np.log(new_probs) + self._default_mistake_counter.get(length + 1))
        self._history_logit += np.log(self._probs[-1][utils.char_to_int(letter)] + 1e-3)
        self._probs.append(new_probs)
        return new_probs

    def get_best_char(self, probs):
        return utils.int_to_char(np.argmax(probs))

    def get_best_hypo(self):
        return self._input_history

    def get_alternatives(self):
        alternatives = []
        for position in range(message_size):
            for letter in range(utils.NUM_SYMBOLS):
                hypo = self._input_history[0:position] + utils.int_to_char(letter)
                if not self._input_history.startswith(hypo):
                    alternatives.append((self._break_logits[position][letter], hypo))
        alternatives = sorted(alternatives, key=lambda (x, y): -x)
        return alternatives

    def _reset(self):
        self._lstm_c = None
        self._lstm_h = None
        self._probs = None
        self._break_logits = []
        self._history_logit = 0
        self._input_history = ''

class DefaultMistakeCounter(object):
    def __init__(self):
        probs = [1] * (message_size + 1)
        with open('model/first-mistake-statistics') as reader:
            for i, line in enumerate(reader):
                probs[i] = float(line) + 1
        for i in range(message_size):
            probs[message_size - i - 1] += probs[message_size - i]
        self._logits = [np.log(probs[-1] / p) for p in probs]

    def get(self, position):
        return self._logits[position]

# TODO: Algorithm is not perfect, it can check the same hypo muttiple times
class HypoSearcher(NetworkAutomata):
    def __init__(self, verbose=True):
        NetworkAutomata.__init__(self)
        self.verbose = verbose

    def search(self, token, num_attempts=100):
        if len(token) > message_size:
            token = token[0:message_size]
        if self.verbose:
            print '\n\nsearching for ', token, '...'
        attempt = 0
        prefixes = [(self._default_mistake_counter.get(0), '')]
        checked_prefixes = []

        while len(prefixes) > 0 and attempt < num_attempts:
            attempt += 1
            prefix = prefixes[0][1]

            # 1. check if there is a good match by default
            hypo, decompressed = self.__get_hypos_from_prefix(token, prefix)
            if len(decompressed) > 0:
                if self.verbose:
                    print 'found something: ', decompressed, '...'
                return '|'.join(decompressed)

            # 2. find max coincided prefix from hypo
            max_coincided_length, best_hypos = self.__get_max_coincided_prefix(token, hypo)
            if len(best_hypos) > 0:
                if self.verbose:
                    print 'found by prefix: ', best_hypos, '...'
                return '|'.join(best_hypos)

            # 3. add hypos
            prefixes.extend(self.get_alternatives())
            prefixes = filter(lambda (x, y): not y.startswith(hypo[0:max_coincided_length]), prefixes)
            checked_prefixes.append(prefix)
            prefixes = filter(lambda (x, y): not y in checked_prefixes, prefixes)
            prefixes = sorted(prefixes, key=lambda (x, y): -x)
        return ''

    def __get_hypos_from_prefix(self, token, prefix):
        probs = self.encode(token)
        for i in range(message_size):
            letter = prefix[i] if i < len(prefix) else self.get_best_char(probs)
            probs = self.apply(letter)
        best_hypo =  self.get_best_hypo()
        if self.verbose:
            print 'trying hypo "' + best_hypo.strip() + '" ...'
        decompressed = cpp_bindings.decompress(best_hypo)
        return best_hypo, decompressed

    def __get_max_coincided_prefix(self, token, hypo):
        hypo = hypo.strip()
        if len(hypo) == 0:
            return 0, []
        for i in range(1, len(hypo)):
            found = cpp_bindings.find_by_prefix(hypo[0:i], 20)
            if len(found) == 0:
                return i - 1, []
            elif len(found) < 20:
                best_hypos = []
                best_levenstein = 4
                for h in found:
                    new_levenstein = cpp_bindings.levenstein(h + ' ' * (100 - len(h)), token + ' ' * (100 - len(token)))
                    if new_levenstein < best_levenstein:
                        best_hypos = [h]
                        best_levenstein = new_levenstein
                    elif new_levenstein == best_levenstein:
                        best_hypos.append(h)
                if len(best_hypos) > 0:
                    return -1, best_hypos
                return i - 1, []
        return len(hypo) - 1, []


counter = 0
mistake_counter = 0
def basic_productivity_check():
    hypo_searcher = HypoSearcher(verbose=False)
    def check(input, expected_output):
        global counter, mistake_counter
        hypos = hypo_searcher.search(input)
        passed = expected_output in hypos
        found_something_else =  len(hypos) > 0
        counter += 1
        if not passed: mistake_counter += 1
        to_print = input + ' -> ' + expected_output
        to_print += ' ' * (60 - len(to_print))
        print to_print, 'PASSED' if passed else 'SMTH ELSE' if found_something_else else 'FAILED', '...'

    check('bridcage walk', 'bridge walk')
    check('krelist louwenstraat', 'krelis louwenstraat')
    check('kfkastrasse', 'kafkastrasse')
    check('bkerstreet', 'baker street')
    check('viaarno', 'via arno')
    check('ia arno', 'via arno')
    check('oosterkstraat', 'oosterstraat')
    check('oosterkstraat', 'oosterstraat')
    check('alae cresei', 'alea cresei')
    check('aleacresei', 'alea cresei')
    check('aleacresei', 'alea cresei')
    check('bulvardul erorol', 'bulevardul eroilor')
    check('bulevardulerolor', 'bulevardul eroilor')
    check('bulvardul eroilor', 'bulevardul eroilor')
    check('piatata abator', 'piata abator')
    check('piattaabator', 'piata abator')
    check('strade amman', 'strada amman')
    check('stradedridu', 'strada dridu')
    check('strade zgravi', 'strada zugravi')
    check('strada tejen', 'strada teajen')
    check('arodrome', 'aerodrome')
    check('are du beau marais', 'aire du beau marais')
    check('are dubeau marais', 'aire du beau marais')
    check('airee du graier', 'aire du granier')
    check('biae des trepasses', 'baie des trepasses')
    check('bosnet', 'boisnet')
    check('borney bas', 'bornay bas')
    check('bordehaute', 'borde haute')
    check('bulervar augusting', 'boulevard augustin')
    check('piethenkade', 'piet henkade')
    check('hbbemstraat', 'hobbemastraat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test neural network for typos correction')
    parser.add_argument('-c', '--command',             type=str,   help='command to process',            required=True, choices=['train', 'play', 'test', 'listen', 'check'])
    parser.add_argument('-i', '--input-folder',        type=str,   help='folder with tokens',            default='model/update-regions')
    parser.add_argument('-m', '--message-size',        type=int,   help='length of each token in batch', default=25)
    parser.add_argument('-b', '--batch-size',          type=int,   help='number of tokens in batch',     default=512)
    parser.add_argument('-f', '--model-file',          type=str,   help='file with binary model',        default=None)
    parser.add_argument('-n', '--ngrams-file',         type=str,   help='file with ngrams model',        default='model/ngrams')
    parser.add_argument('-p', '--mistake-probability', type=float, help='mistake probability',           default=0.2)
    args = parser.parse_args()
    message_size = args.message_size
    batch_size = args.batch_size
    model_file = args.model_file
    model_file = model_file if not model_file is None else 'model/model-1/model'

    cpp_bindings.generate_cpp_bindings(ngrams_file=args.ngrams_file,
                                       update_regions_folder=args.input_folder,
                                       mistake_probability=args.mistake_probability,
                                       message_size=args.message_size)

    if args.command == 'train':
        network = Network()
        network.train()
    elif args.command == 'play':
        hypo_searcher = HypoSearcher()
        while True:
            token = raw_input("Input something: ")
            hypo_searcher.search(token)
            print ''
    elif args.command == 'test':
        first_mistake_statistics = np.zeros(message_size + 1)
        automata = NetworkAutomata()
        clean_test_batch, contaminated_test_batch = cpp_bindings.generate_random_batch(test_batch_size,
                                                                                       use_one_update_region=False)
        for _ in range(test_batch_size):
            if _ % 100 == 0: print _
            clean_token = utils.numpy_to_string(clean_test_batch[_, :])
            contaminated_token = utils.numpy_to_string(contaminated_test_batch[_, :])
            probs = automata.encode(contaminated_token)
            for i in range(message_size):
                letter = automata.get_best_char(probs)
                if letter != clean_token[i]:
                    first_mistake_statistics[i] += 1
                    break
                if i + 1 == message_size:
                    first_mistake_statistics[-1] += 1
                probs = automata.apply(letter)
        with open('model/first-mistake-statistics', 'w') as writer:
            writer.write('\n'.join([str(_) for _ in first_mistake_statistics]))
    elif args.command == 'listen':
        hypo_searcher = HypoSearcher()
        communicator = protobuf_talker.ProtobufTalker()
        print 'listening ...'
        while True:
            address, received = communicator.receive()
            to_send = hypo_searcher.search(received)
            communicator.send(address, to_send)
    elif args.command == 'check':
        basic_productivity_check()
    else:
        raise ValueError(args.command)

