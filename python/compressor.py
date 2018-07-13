import argparse, ctypes
import numpy as np
from utils import numpy_to_string

library = ctypes.cdll.LoadLibrary('../build/libbatch-generator.so')

class Compressor(object):
    def __init__(self, dictionary_file):
        library.create_compressor(ctypes.c_char_p(dictionary_file))

    def decompress(self, token):
        decompressed = ' ' * 1024
        library.decompress(ctypes.c_char_p(token), ctypes.c_char_p(decompressed))
        return decompressed.strip().split('|')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of batch generator')
    parser.add_argument('-i', '--input-file', type=str, help='file with prefix tree', default='model/dictionary')
    args = parser.parse_args()

    compressor = Compressor(args.input_file)
    print compressor.decompress('osterdokstrat')
    print compressor.decompress('amsterdamveg')
