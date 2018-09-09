import numpy as np
import ctypes
import unicodedata

library = ctypes.cdll.LoadLibrary('../build/python-bindings/libpython-bindings.so')

A_INT = np.int32(ord('a'))
Z_INT = np.int32(ord('z'))
INT_0 = np.int32(ord('0'))
INT_9 = np.int32(ord('9'))
NUM_ALPHAS = Z_INT - A_INT + 1
NUM_DIGITS = 10
SPACE_INT = NUM_ALPHAS + NUM_DIGITS
SEPARATOR_INT = SPACE_INT + 1
EFFECTIVE_NUM_SYMBOLS = SEPARATOR_INT + 1
NUM_SYMBOLS = 64

def acceptable(letter): 
    return letter == ' ' or letter == '|' or \
           (A_INT <= ord(letter) and ord(letter) <= Z_INT) or \
           (INT_0 <= ord(letter) and ord(letter) <= INT_9)

def char_to_int(letter):
    assert acceptable(letter)
    return SPACE_INT if letter == ' ' \
        else SEPARATOR_INT if letter == '|' \
        else NUM_ALPHAS + ord(letter) - INT_0 if ord(letter) <= INT_9 \
        else ord(letter) - A_INT

def int_to_char(number):
    assert 0 <= number and number <= SEPARATOR_INT
    return ' ' if number == SPACE_INT \
        else '|' if number == SEPARATOR_INT \
        else chr(A_INT + number) if number < NUM_ALPHAS \
        else chr(INT_0 + number - NUM_ALPHAS)

def numpy_to_string(array):
    return ''.join([int_to_char(_) for _ in array])

def string_to_numpy(string):
    return np.array([char_to_int(letter) for letter in string], dtype=np.int32)

def strip_accents(input):
    return unicodedata.normalize('NFKD', input.decode('utf-8')).encode('ASCII', 'ignore')


