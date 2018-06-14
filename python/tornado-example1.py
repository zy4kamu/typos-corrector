import os
import tornado.ioloop
import tornado.web
from  tornado import web

__author__ = 'gvincent'

root = os.path.dirname(__file__)
port = 8888

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            name = self.get_argument('name', '')
        except AssertionError:
            pass
        try:
            with open(os.path.join(root, 'index.html')) as f:
                if len(name) > 0:
                    self.write(f.read() + "\n" + name)
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
