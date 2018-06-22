import os, time
import tornado.ioloop
import tornado.web
from  tornado import web
import network

root = os.path.dirname(__file__)
port = 8889
automata = network.Network()
automata.read_all_required_tensors_from_file()

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            token = self.get_argument('street', '')
        except AssertionError:
            pass
        try:
            with open(os.path.join(root, 'index.html')) as f:
                self.write(f.read())
                if len(token) > 0:
                    prediction = automata.make_prediction_for_token(token)
                    self.write(prediction)
        except IOError as e:
            self.write("404: Not Found")

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/(.*)", web.StaticFileHandler, dict(path=root)),
    ])

if __name__ == '__main__':
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()
