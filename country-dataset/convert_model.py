import sys
import numpy as np

input_file = sys.argv[1]
output_file = sys.argv[2]

weights = []
with open(input_file) as reader:
    for _ in range(10): reader.readline()
    for line in reader:
        weights.append(float(line.strip().split(':')[1]))

data = np.asarray(weights, dtype=np.float32)
data.tofile(output_file)
