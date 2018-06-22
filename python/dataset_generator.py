import argparse, ctypes
import numpy as np
from utils import numpy_to_string

library = ctypes.cdll.LoadLibrary('../build/libbatch-generator.so')

class DataSetGenerator(object): 
    def __init__(self, dictionary_file, message_size, mistake_probability):
        library.create_random_batch_generator(ctypes.c_char_p(dictionary_file),
                                              ctypes.c_double(mistake_probability))
        self.message_size = message_size

    def next(self, batch_size):
        clean = np.empty([batch_size, self.message_size], dtype=np.int32)
        contaminated = np.empty([batch_size, self.message_size], dtype=np.int32)
        library.generate_random_batch(clean.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 
                                      contaminated.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                      ctypes.c_size_t(self.message_size), 
                                      ctypes.c_size_t(batch_size))
        return clean, contaminated

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of batch generator')
    parser.add_argument('-i', '--input-file', type=str, help='file with prefix tree', default='model/dictionary')
    parser.add_argument('-m', '--message-size', type=int, help='length of each token in batch', default=30)
    parser.add_argument('-b', '--batch-size', type=int, help='number of tokens in batch', default=300)
    parser.add_argument('-p', '--mistake-probability', type=int, help='probability to make a mistake', 
                        default=0.2)
    args = parser.parse_args()

    generator = DataSetGenerator(args.input_file, args.message_size, args.mistake_probability)
    clean, contaminated = generator.next(args.batch_size)
    for i in range(args.batch_size):
        print i, numpy_to_string(clean[i, :]), numpy_to_string(contaminated[i, :])
