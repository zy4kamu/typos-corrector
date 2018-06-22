import numpy as np
import ctypes

library = ctypes.cdll.LoadLibrary('../build/libbatch-generator.so')

A_INT = np.int32(ord('a'))
Z_INT = np.int32(ord('z'))
SPACE_INT = Z_INT - A_INT + 1
NUM_SYMBOLS = SPACE_INT + 1

def acceptable(letter): 
    return letter == ' ' or ord('a') <= ord(letter) or ord(letter) <= ord('z')

def char_to_int(letter):
    assert acceptable(letter)
    return SPACE_INT if letter == ' ' else ord(letter) - A_INT

def int_to_char(number):
    assert 0 <= number and number <= SPACE_INT
    return ' ' if number == SPACE_INT else chr(A_INT + number)

def numpy_to_string(array):
    return ''.join([int_to_char(_) for _ in array])

def string_to_numpy(string):
    return np.array([char_to_int(letter) for letter in string], dtype=np.int32)

def levenstein(first_message, second_message):
    assert type(first_message) == str
    assert type(second_message) == str
    assert len(first_message) == len(second_message)
    func = library.levenstein
    func.restype = ctypes.c_size_t
    return func(ctypes.c_char_p(first_message), ctypes.c_char_p(second_message), ctypes.c_size_t(len(first_message)))

if __name__ == '__main__':
    assert levenstein('abcd', 'bcde') == 2
    assert levenstein('abcdd', 'bcded') == 2
    assert levenstein('hello', 'dello') == 1
    assert levenstein('amsterdam', 'masterdam') == 2
    assert levenstein('amsterdamm', 'masterdamk') == 3
