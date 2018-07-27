import ctypes
import numpy as np

_library = ctypes.cdll.LoadLibrary('../build/libpython-bindings.so')


def generate_cpp_bindings(update_regions_folder='model/update-regions', mistake_probability=0.2):
    _set_update_regions_folder(update_regions_folder)
    _create_update_regions_set()
    _create_contaminator(mistake_probability)
    _create_compressor()
    _create_random_batch_generator()


def decompress(token):
    decompressed = ' ' * 1024
    _library.decompress(ctypes.c_char_p(token.strip()), ctypes.c_char_p(decompressed))
    return filter(lambda x: len(x) > 0, decompressed.strip().split('|'))


def find_by_prefix(token, max_number):
    decompressed = ' ' * 100000
    _library.find_by_prefix(ctypes.c_char_p(token.strip()), ctypes.c_size_t(max_number),
                            ctypes.c_char_p(decompressed))
    return filter(lambda x: len(x) > 0, decompressed.strip().split('|'))


def generate_random_batch(batch_size, message_size, use_one_update_region=True):
    clean = np.empty([batch_size, message_size], dtype=np.int32)
    contaminated = np.empty([batch_size, message_size], dtype=np.int32)
    if use_one_update_region:
        function = _library.generate_random_batch_on_one_update_region
        function.restype = ctypes.c_int32
        update_region_id = function(clean.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                    contaminated.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                    ctypes.c_size_t(message_size),
                                    ctypes.c_size_t(batch_size))
        return update_region_id, clean, contaminated
    else:
        function = _library.generate_random_batch_on_all_update_regions
        function.restype = ctypes.c_int32
        function(clean.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                 contaminated.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                 ctypes.c_size_t(message_size),
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


def _set_update_regions_folder(update_regions_folder):
    _library.set_update_regions_folder(ctypes.c_char_p(update_regions_folder))


def _create_update_regions_set():
    _library.create_update_regions_set()


def _create_contaminator(mistake_probability):
    _library.create_contaminator(ctypes.c_double(mistake_probability))


def _create_compressor():
    _library.create_compressor()


def _create_random_batch_generator():
    _library.create_random_batch_generator()


# Examples


if __name__ == '__main__':
    assert levenstein('abcd', 'bcde') == 2
    assert levenstein('abcdd', 'bcded') == 2
    assert levenstein('hello', 'dello') == 1
    assert levenstein('amsterdam', 'masterdam') == 2
    assert levenstein('amsterdamm', 'masterdamk') == 3

    generate_cpp_bindings('model/update-regions', 0.2)
    print find_by_prefix('krelis', 20)


