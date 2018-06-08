import ctypes, argparse

library = ctypes.cdll.LoadLibrary('../build/libtypos-corrector-helper.so')
MAX_TOKEN_LENGTH = 256

class PrefixTree(object):
    def __init__(self, file_name=None):
        if file_name is None:
            library.create()
        else: 
            library.create_from_file(ctypes.c_char_p(file_name), len(file_name))

    def match(self, token):
        assert type(token) == str
        function = library.match
        function.restype = ctypes.c_size_t
        return int(function(ctypes.c_char_p(token), len(token)))

    def add(self, token):
        assert type(token) == str
        library.add(ctypes.c_char_p(token), len(token))

    def __del_(self):
        library.destroy()

    def generate(self):
        token = ' ' * MAX_TOKEN_LENGTH
        library.generate(ctypes.c_char_p(token))
        return token.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test prefix tree')
    parser.add_argument('-f', '--file', type=str, help='file with prefix tree', default='')
    args = parser.parse_args()
    if args.file == '':
        tree = PrefixTree()
        tree.add('hello')
        tree.add('this')
        tree.add('is')
        tree.add('prefix')
        tree.add('tree')

        assert tree.match('help') == 3
        assert tree.match('hell') == 4
        assert tree.match('hrll') == 1
        assert tree.match('ist') == 2
        assert tree.match('is') == 2
        assert tree.match('ls') == 0

        print 'all assetions passed'
        print 'random generation: ',
        for _ in range(10):
            print tree.generate(), ' ',
    else:
        tree = PrefixTree(args.file)
        assert tree.match('help') == 4
        assert tree.match('helpp') == 4
        assert tree.match('amsterdam') == 9
        assert tree.match('0') == 0

        print 'all assetions passed'
        print 'random generation: ',
        for _ in range(10):
            print tree.generate(), ' ',

