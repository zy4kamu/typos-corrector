# -*- coding: utf-8 -*-
import os, unicodedata, shutil

import utils


input_file = os.path.join(os.environ['HOME'], 'git-repos/typos-corrector/dataset/preprocessed/data')
output_folder = os.path.join(os.environ['HOME'], 'git-repos/typos-corrector/dataset/all')
output_transitions_file = os.path.join(output_folder, 'data')
output_names_dict_file = os.path.join(os.environ['HOME'], 'git-repos/typos-corrector/dataset/all/names')
by_country_folder = os.path.join(os.environ['HOME'], 'git-repos/typos-corrector/dataset/by-country')

# Functions to create names_dict and transitions_dict

names_dict = {}
transitions_dict = {}


def strip_accents(input):
    return unicodedata.normalize('NFKD', input.decode('utf-8'))\
        .replace(u'ł', u'l' ) \
        .replace(u'ß', u'ss') \
        .replace(u'đ', u'd' ) \
        .replace(u'ı', u'i' ) \
        .replace(u'Ł', u'l' ) \
        .replace(u'Ø', u'o' ) \
        .replace(u'ð', u'o' ) \
        .replace(u'ø', u'o' ) \
        .replace(u'Đ', u'd' ) \
        .replace(u'Æ', u'ae') \
        .replace(u'œ', u'oe') \
        .replace(u'Þ', u'p' ) \
        .replace(u'þ', u'p' ) \
        .replace(u'æ', u'ae') \
        .encode('ASCII', 'ignore')


def get_from_names_dict(key):
    if key in names_dict:
        return names_dict[key]
    else:
        to_add = len(names_dict)
        names_dict[key] = to_add
        return to_add


def get_from_transitions_dict(dic, key):
    if key in dic:
        return dic[key]
    else:
        dic[key] = {}
        return dic[key]


def process_buffer(buffer):
    for item in buffer:
        if len(item) == 0 or item[-1] == '|' or len(item.split('|')) != 2 or item[0] == '|':
            print 'bad buffer detected:', buffer
            return

    dic = transitions_dict
    for line in buffer:
        if line[0] == 'h':
            hsns = line[2:].split(' ')
            hsns = filter(lambda _: len(_) > 0, hsns)
            for hsn in hsns:
                hsn = 'h|' + hsn
                index = get_from_names_dict(hsn)
                get_from_transitions_dict(dic, index)
        else:
            index = get_from_names_dict(line)
            dic = get_from_transitions_dict(dic, index)

# Functions to create output file

def print_names_dict():
    def type_to_int(type):
        if type == 'h': return 101
        if type == 's': return 100
        return int(type)
    with open(output_names_dict_file, 'w') as writer:
        data = names_dict.items()
        data = sorted(data, key=lambda (key, value):  len(data) * type_to_int(key.split('|')[0]) + value)
        for key, value in data:
            type, entity = key.split('|')
            writer.write('{}|{}|{}\n'.format(type, value, entity))

def print_transitions_dict():
    def print_dic(writer, dic, prefix):
        for key, value in dic.iteritems():
            if len(value) == 0:
                writer.write(' '.join([prefix, str(key)]).strip())
                writer.write('\n')
            else:
                print_dic(writer, value, ' '.join([prefix, str(key)]))
    with open(output_transitions_file, 'a') as writer:
        print_dic(writer, transitions_dict, '')

# Main input function

def create_all_dataset_folder():
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    buffer = []
    counter = 0
    with open(input_file) as reader:
        for line in reader:
            line = strip_accents(line.strip().lower()).lower()
            if line == 'arc end':
                process_buffer(buffer)
                buffer = []
                counter += 1
                if counter % 100000 == 0:
                    print 'evaluated', counter, 'examples ...'
                    print_transitions_dict()
                    transitions_dict.clear()
            else:
                splitted = line.split(' ')
                line = splitted[0] + '|' + ' '.join(splitted[1:])
                line = line.strip()
                line = ''.join([ch for ch in line if utils.acceptable(ch)])
                if line[-1] != '|': # empty entities can easily happen for Greece and Russia since they don't use latinic
                    buffer.append(line)
    print_transitions_dict()
    print_names_dict()
    os.system('sort -n {} | uniq > tmp'.format(output_transitions_file))
    os.system('mv tmp {}'.format(output_transitions_file))

def separate_by_country():
    if os.path.exists(by_country_folder):
        shutil.rmtree(by_country_folder)
    os.mkdir(by_country_folder)

    # create index_to_country
    index_to_country_writer = {}
    with open(output_names_dict_file) as reader:
        for line in reader:
            type, index, country = line.strip().split('|')
            if type != '3': break
            if not index in index_to_country_writer:
                country_folder = os.path.join(by_country_folder, country)
                os.mkdir(country_folder)
                output_file = os.path.join(country_folder, 'data')
                index_to_country_writer[index] = open(output_file, 'w')

    # iterate over file
    with open(output_transitions_file) as reader:
        for line in reader:
            index = line.split(' ')[0]
            index_to_country_writer[index].write(line)

    # close all writers
    for writer in index_to_country_writer.values():
        writer.close()

def create_ngrams():
    names_dict = {}
    with open(output_names_dict_file) as reader:
        for line in reader:
            type, index, country = line.strip().split('|')
            names_dict[index] = country

    ngram_size = 3
    spaces = (' ' * ngram_size)
    for country in os.listdir(by_country_folder):
        print 'working with', country
        # create ngrams
        ngrams = {}
        with open(os.path.join(by_country_folder, country + '/data')) as reader:
            for line in reader:
                splitted = line.strip().split(' ')
                for index in splitted:
                    token = names_dict[index]
                    token = spaces + token.strip() + spaces
                    for i in range(len(token) - ngram_size):
                        ngram = token[i:i + ngram_size]
                        next_char = token[i + ngram_size]
                        if ngrams.get(ngram) is None:
                            ngrams[ngram] = {}
                        chars = ngrams[ngram]
                        chars[next_char] = chars.get(next_char, 0) + 1
        # print ngrams
        with open(os.path.join(by_country_folder, country + '/ngrams'), 'w') as writer:
            for k1, v1 in ngrams.iteritems():
                for k2, v2 in v1.iteritems():
                    writer.write('{}|{}|{}\n'.format(k1, k2, v2))

def create_names_symlinks():
    for country in os.listdir(by_country_folder):
        output_link = os.path.join(by_country_folder, country + "/names")
        os.remove(output_link)
        os.symlink(output_names_dict_file, output_link)

def create_common_ngrams_file():
    ngrams = {}
    for country in os.listdir(by_country_folder):
        ngrams_file = os.path.join(by_country_folder, country + "/ngrams")
        with open(ngrams_file) as reader:
            for line in reader:
                key = '|'.join(line.split('|')[:-1])
                value = int(line.split('|')[-1])
                if key in ngrams:
                    ngrams[key] += value
                else:
                    ngrams[key] = value
    ngrams = sorted(ngrams.items())
    with open(os.path.join(output_folder, 'ngrams'), 'w') as writer:
        for key, value in ngrams:
            writer.write('{}|{}\n'.format(key, value))

if __name__ == '__main__':
    """
    print 'step 0: creating {} from {}'.format(input_file, output_folder)
    create_all_dataset_folder()
    print 'step 1: separate {} to {}'.format(output_folder, by_country_folder)
    separate_by_country()
    print 'step 2: creating ngrams'
    create_ngrams()
    print 'step3: creating symlinks for names'
    create_names_symlinks()
    """
    print 'step4: creating common ngrams file'
    create_common_ngrams_file()
