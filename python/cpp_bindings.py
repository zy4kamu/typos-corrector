import ctypes, random
import numpy as np

_library = ctypes.cdll.LoadLibrary('../build/python-bindings/libpython-bindings.so')

MESSAGE_SIZE = None

def generate_cpp_bindings(country, mistake_probability, message_size):
    global MESSAGE_SIZE
    MESSAGE_SIZE = message_size
    _reset()
    _set_dataset_folder('../dataset/by-country/' + country)
    _create_dataset()
    _create_contaminator('../dataset/by-country/{}/ngrams'.format(country), mistake_probability)
    _create_random_batch_generator()


def find_by_prefix(token, max_number):
    decompressed = ' ' * 100000
    _library.find_by_prefix(ctypes.c_char_p(token.strip()), ctypes.c_size_t(max_number),
                            ctypes.c_char_p(decompressed))
    return filter(lambda x: len(x) > 0, decompressed.strip().split('|'))


def generate_random_batch(batch_size):
    clean = np.empty([batch_size, MESSAGE_SIZE], dtype=np.int32)
    contaminated = np.empty([batch_size, MESSAGE_SIZE], dtype=np.int32)
    function = _library.generate_random_batch
    function.restype = ctypes.c_int32
    function(clean.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
             contaminated.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
             ctypes.c_size_t(batch_size))
    return clean, contaminated


def levenstein(first_message, second_message):
    assert type(first_message) == str
    assert type(second_message) == str
    assert len(first_message) == len(second_message)
    func = _library.levenstein
    func.restype = ctypes.c_size_t
    return func(ctypes.c_char_p(first_message), ctypes.c_char_p(second_message), ctypes.c_size_t(len(first_message)))


# Helpers

def _reset():
    _library.reset()


def _set_dataset_folder(dataset_folder):
    _library.set_dataset_folder(ctypes.c_char_p(dataset_folder))

def _create_dataset():
    _library.create_dataset(ctypes.c_size_t(-1))


def _create_contaminator(ngrams_file, mistake_probability):
    _library.create_contaminator(ctypes.c_char_p(ngrams_file), ctypes.c_double(mistake_probability))


def _create_random_batch_generator():
    _library.create_random_batch_generator(ctypes.c_size_t(MESSAGE_SIZE))


# Examples


if __name__ == '__main__':
    assert levenstein('abcd', 'bcde') == 2
    assert levenstein('abcdd', 'bcded') == 2
    assert levenstein('hello', 'dello') == 1
    assert levenstein('amsterdam', 'masterdam') == 2
    assert levenstein('amsterdamm', 'masterdamk') == 3

