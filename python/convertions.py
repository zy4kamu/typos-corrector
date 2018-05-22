import numpy as np

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
