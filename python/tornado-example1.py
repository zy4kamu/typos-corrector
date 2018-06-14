import os, time
import numpy as np
import tornado.ioloop
import tornado.web
from  tornado import web
from prefix_tree import PrefixTree
import tensorflow as tf

from convertions import SPACE_INT, string_to_numpy, NUM_SYMBOLS

model_file = 'model/model-1/model'
message_size = 30
root = os.path.dirname(__file__)
port = 8888
num_hypos = 15

sess = tf.Session()
saver = tf.train.import_meta_graph(model_file + '.meta', clear_devices=True)
saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file + '.meta')))
graph = tf.get_default_graph()
contaminated_tokens = graph.get_tensor_by_name("contaminated_tokens:0")
play_logits = [graph.get_tensor_by_name('play_logit_{}:0'.format(i)) for i in range(message_size)]
prefix_tree = PrefixTree('model/prefix-tree')

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            token = self.get_argument('street', '')
        except AssertionError:
            pass
        try:
            with open(os.path.join(root, 'index.html')) as f:
                if len(token) > 0:
                    self.write(f.read() + "\n")
                    numpy_token = np.ones(message_size, dtype=np.int32) * SPACE_INT
                    numpy_token[0:len(token)] = string_to_numpy(token)
                    numpy_token = numpy_token.reshape((-1, message_size))
                    start_time = time.time()
                    predictions, = sess.run([play_logits], feed_dict={contaminated_tokens: numpy_token})
                    tensorlfow_time = time.time() - start_time
                    np_predictions = np.empty(dtype=np.double, shape=[len(predictions), NUM_SYMBOLS])
                    for i, prediction in enumerate(predictions):
                        np_predictions[i, :] = predictions[i]
                    start_time = time.time()
                    viterbi_tokens, viterbi_logits = prefix_tree.viterbi(np_predictions, num_hypos)
                    for t, l in zip(viterbi_tokens[0:10], viterbi_logits[0:10]):
                        self.write("%.2f" % l)
                        self.write('&nbsp')
                        self.write(t)
                        self.write('<br/>')
                    self.write('tensorflow time: {}'.format(tensorlfow_time))
                    self.write('<br/>')
                    self.write('viterbi time: {}'.format(time.time() - start_time))
                else:
                    self.write(f.read())
        except IOError as e:
            self.write("404: Not Found")

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/(.*)", web.StaticFileHandler, dict(path=root)),
    ])

if __name__ == '__main__':
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()
