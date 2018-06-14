import ctypes, argparse
import numpy as np
from convertions import NUM_SYMBOLS, char_to_int

library = ctypes.cdll.LoadLibrary('../build/libprefix-tree.so')
MAX_TOKEN_LENGTH = 256

class PrefixTree(object):
    def __init__(self, file_name):
        library.create_from_file(ctypes.c_char_p(file_name), len(file_name))

    def match(self, token):
        assert type(token) == str
        function = library.match
        function.restype = ctypes.c_size_t
        return int(function(ctypes.c_char_p(token), len(token)))

    def __del_(self):
        library.destroy()

    def viterbi(self, logits, num_hypos):
        result_string = ' ' * 256 * num_hypos
        result_logits = np.empty(num_hypos, dtype=np.double)
        library.viterbi(logits.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        logits.shape[0],
                        num_hypos,
                        ctypes.c_char_p(result_string),
                        result_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return result_string.strip().split('$'), result_logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test prefix tree')
    parser.add_argument('-f', '--file', type=str, help='file with prefix tree', required=True)
    args = parser.parse_args()
    tree = PrefixTree(args.file)
    print tree.match('oosterdok')

    logits = 0.1 * np.ones(dtype=np.double, shape=(9, NUM_SYMBOLS))
    for i, letter in enumerate('oosterdok'):
        logits[i, char_to_int(letter)] = 1
    result_tokens, result_logits = tree.viterbi(logits, 10)
    for t,l in zip(result_tokens, result_logits):
        print t, l
