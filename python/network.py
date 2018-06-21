import argparse, os
import tensorflow as tf
import numpy as np
import convertions
from dataset_generator import DataSetGenerator

command             = None
input_folder        = None
message_size        = None
batch_size          = None
model_file          = None
num_hypos           = 1000
test_num_iterations = 2500
test_batch_size     = 10000
lstm_size           = 512

class Network(object):
    def initialize_for_training(self):
        with tf.device("/device:GPU:0"):
            # inputs
            self.clean_tokens = tf.placeholder(tf.int32, [None, message_size], name='clean_tokens')
            self.contaminated_tokens = tf.placeholder(tf.int32, [None, message_size], name='contaminated_tokens')

            clean_one_hot_embedding = tf.one_hot(self.clean_tokens, convertions.NUM_SYMBOLS)
            char_embedding_matrix = tf.Variable(tf.random_uniform([convertions.NUM_SYMBOLS, convertions.NUM_SYMBOLS], -1.0, 1.0))
            contaminated_embedding = tf.nn.embedding_lookup(char_embedding_matrix, self.contaminated_tokens)
            clean_embedding = tf.nn.embedding_lookup(char_embedding_matrix, self.clean_tokens)

            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            hidden_layer_weights = tf.Variable(tf.truncated_normal([lstm_size, convertions.NUM_SYMBOLS]))
            hidden_layer_bias  = tf.Variable(tf.truncated_normal([convertions.NUM_SYMBOLS]))

            def create_architecture(batch_size):
                logits = []
                state = lstm.zero_state(batch_size, tf.float32)
                for i in range(message_size):
                    output, state = lstm(contaminated_embedding[:, message_size - i - 1, :], state)
                for i in range(message_size):
                    logits.append(tf.matmul(output, hidden_layer_weights) + hidden_layer_bias)
                    output, state = lstm(clean_embedding[:, i, :], state)
                return logits

            # training
            self.train_logits = create_architecture(batch_size)
            self.total_loss = None
            for i in range(message_size):
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.train_logits[i], labels=clean_one_hot_embedding[:, i, :])
                loss = tf.reduce_sum(loss)
                if self.total_loss is None: self.total_loss = loss
                else: self.total_loss += loss
            self.optimizer = tf.train.AdamOptimizer().minimize(self.total_loss)

            # testing
            self.test_logits = create_architecture(test_batch_size)

            # initial play state
            initial_play_state = lstm.zero_state(1, tf.float32)
            for i in range(message_size):
                output, initial_play_state = lstm(contaminated_embedding[:, message_size - i - 1, :], initial_play_state)
            tf.identity(initial_play_state.c, 'initial_play_state_c')
            tf.identity(initial_play_state.h, 'initial_play_state_h')
            tf.identity(tf.matmul(output, hidden_layer_weights) + hidden_layer_bias, 'initial_logits')

            # apply lstm cell
            apply_input_state_c = tf.placeholder(dtype=tf.float32, shape=(1, lstm_size), name='apply_input_state_c')
            apply_input_state_h = tf.placeholder(dtype=tf.float32, shape=(1, lstm_size), name='apply_input_state_h')
            apply_input_char = tf.placeholder(dtype=np.int32, shape=(1,), name='apply_input_char')
            apply_input_state = tf.contrib.rnn.LSTMStateTuple(apply_input_state_c, apply_input_state_h)
            apply_input_embedding = tf.nn.embedding_lookup(char_embedding_matrix, apply_input_char)
            apply_output_logits, apply_output_state = lstm(apply_input_embedding, apply_input_state)
            tf.identity(tf.matmul(apply_output_logits, hidden_layer_weights) + hidden_layer_bias, 'after_apply_logits')
            tf.identity(apply_output_state.c, 'after_apply_state_c')
            tf.identity(apply_output_state.h, 'after_apply_state_h')

    def train(self):
        batch_generator = DataSetGenerator(dictionary_file='model/dictionary', message_size=message_size, mistake_probability=0.2)
        clean_test_batch, contaminated_test_batch = batch_generator.next(test_batch_size)
        counter = 0
        train_num_correct = 0
        train_num_letters = 0

        initializer = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(initializer)
            while True:
                # update gradient
                clean, contaminated = batch_generator.next(batch_size)
                _, predictions, l = sess.run([self.optimizer, self.train_logits, self.total_loss],
                    feed_dict={self.clean_tokens:clean, self.contaminated_tokens:contaminated})

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
                    predictions, = sess.run([self.test_logits], feed_dict={ self.contaminated_tokens:contaminated_test_batch,
                                                                            self.clean_tokens:clean_test_batch })
                    test_num_letters += test_batch_size * message_size
                    for i in range(message_size):
                        test_num_correct += np.sum((np.argmax(predictions[i], axis=1) == clean_test_batch[:, i]))
                        dummy_num_correct += np.sum(clean_test_batch[:, i] == contaminated_test_batch[:, i])
                    print 'test: {} correct of {}; accuracy = {}'.format(test_num_correct, test_num_letters,
                                                                         float(test_num_correct) / float(test_num_letters))
                    print 'dummy: {} correct of {}; accuracy = {}'.format(dummy_num_correct, test_num_letters,
                                                                          float(dummy_num_correct) / float(test_num_letters))
                    print ''

    def initializea_for_testing(self):
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

    def get_best_token(self, token):
        # convert token to numpy array
        token += ' ' * (message_size - len(token))
        numpy_token = np.ones(message_size, dtype=np.int32) * convertions.SPACE_INT
        numpy_token[0:len(token)] = convertions.string_to_numpy(token)
        numpy_token = numpy_token.reshape((-1, message_size))

        # create intial state (encode)
        lstm_c, lstm_h, logits = self.sess.run(
            [self.initial_play_state_c, self.initial_play_state_h, self.initial_logits],
            feed_dict={self.contaminated_tokens:numpy_token})

        # pass over token and decode token (decode)
        prefix = ''
        for i in range(message_size):
            best_char_index = np.argmax(logits[0, :])
            numpy_char = np.ones(dtype=np.int32, shape=(1,)) * best_char_index
            lstm_c, lstm_h, logits = self.sess.run(
                [self.apply_output_state_c, self.apply_output_state_h, self.apply_output_logits],
                feed_dict={self.apply_input_state_c:lstm_c, self.apply_input_state_h:lstm_h, self.apply_input_char:numpy_char})
            prefix += convertions.int_to_char(best_char_index)
        print prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test neural network for typos correction')
    parser.add_argument('-c', '--command',          type=str, help='command to process',            required=True, choices=['train', 'play'])
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
        network.initialize_for_training()
        network.train()
    elif command == 'play':
        network.initializea_for_testing()
        while True:
            token = raw_input("Input something: ")
            network.get_best_token(token)
    else:
        raise ValueError(command)
