import sys 

predictions_file = sys.argv[1]
real_file = sys.argv[2]

counter = 0
correct_counter = 0
rank_counter = 0
with open(predictions_file) as predicted_reader, open(real_file) as real_reader:
    for line in predicted_reader:
        counter += 1
        predicted = map(lambda x: (int(x.split(':')[0]), float(x.split(':')[1])), 
                        line.strip().split(' '))
        predicted = sorted(predicted, key=lambda (x,y): -y)
        predicted = map(lambda (x, y): x, predicted)

        line = real_reader.readline()
        real = line.split(' ')[0].strip()

        if int(real) == int(predicted[0]): correct_counter += 1
        rank_counter += predicted.index(int(real))

print correct_counter, 'of', counter, 'accuracy =', float(correct_counter) / float(counter)
print 'average rank =', float(rank_counter) / float(counter)
