import os


input_directory = 'model/update-regions'
ngrams = {}
ngram_size = 3

spaces = (' ' * ngram_size)

for file in os.listdir(input_directory):
    full_path = os.path.join(input_directory, file)
    with open(full_path) as reader:
        for token in reader:
            token = spaces + token.strip() + spaces
            for i in range(len(token) - ngram_size):
                ngram = token[i:i + ngram_size]
                next_char = token[i + ngram_size]
                if ngrams.get(ngram) is None:
                    ngrams[ngram] = {}
                chars = ngrams[ngram]
                chars[next_char] = chars.get(next_char, 0) + 1

with open('model/ngrams', 'w') as writer:
    for k1, v1 in ngrams.iteritems():
        for k2, v2 in v1.iteritems():
            writer.write('{}|{}|{}\n'.format(k1, k2, v2))
    


