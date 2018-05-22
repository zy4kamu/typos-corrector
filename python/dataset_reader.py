import argparse, os, sys 
import numpy as np
from convertions import numpy_to_string

class DataSetReader(object): 
    def __init__(self, input_folder, message_size, batch_size, first_batch=0, last_batch=sys.maxint):
        self.input_folder = input_folder
        self.message_size = message_size
        self.batch_size = batch_size
        self.epoch = 0
        self.first_batch = first_batch
        self.last_batch = last_batch

    def __iter__(self):
        input_folder = self.input_folder
        message_size = self.message_size
        batch_size = self.batch_size
        first_batch = self.first_batch
        last_batch = self.last_batch

        num_files = len(os.listdir(input_folder)) / 2
        while True:
            for i in range(first_batch, min(num_files, last_batch)): 
                clean_file = os.path.join(input_folder, 'clean-{}'.format(i))
                clean = np.fromfile(clean_file, dtype=np.int32).reshape((batch_size, message_size))

                contaminated_file = os.path.join(input_folder, 'contaminated-{}'.format(i))
                contaminated = np.fromfile(contaminated_file, dtype=np.int32).reshape(batch_size, message_size)

                yield clean, contaminated
            self.epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of iterating over batch folder')
    parser.add_argument('-i', '--input-folder', type=str, help='folder with binary batches', required=True)
    parser.add_argument('-m', '--message-size', type=int, help='length of each token in batch', required=True)
    parser.add_argument('-b', '--batch-size', type=int, help='number of tokens in batch', required=True)
    args = parser.parse_args()

    reader = DataSetReader(args.input_folder, args.message_size, args.batch_size)
    for clean, contaminated in reader:
        print numpy_to_string(clean[0, :]), numpy_to_string(contaminated[0, :])
        if reader.epoch == 1: 
            exit(0)

