# -*- coding: utf-8 -*-
import os, unicodedata, shutil

import utils
import cpp_bindings


input_file = os.path.join(os.environ['HOME'], 'git-repos/typos-corrector/dataset/preprocessed/data')
output_folder = os.path.join(os.environ['HOME'], 'git-repos/typos-corrector/dataset/all')
output_transitions_file = os.path.join(output_folder, 'data')
output_names_dict_file = os.path.join(os.environ['HOME'], 'git-repos/typos-corrector/dataset/all/names')
by_country_state_folder = os.path.join(os.environ['HOME'], 'git-repos/typos-corrector/dataset/by-country')

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

def separate_by_country_state(group_dict):
    if os.path.exists(by_country_state_folder):
        shutil.rmtree(by_country_state_folder)
    os.mkdir(by_country_state_folder)

    # create index_to_country_state
    index_to_country_state = {}
    with open(output_names_dict_file) as reader:
        for line in reader:
            type, index, country_state = line.strip().split('|')
            if type != '3' and type != '4': break
            index_to_country_state[index] = (type, groups_dict.get(country_state, country_state))

    # iterate over file
    writers = {}
    with open(output_transitions_file) as reader:
        for line in reader:
            country = 'none-country'
            state = 'none-state'
            for index in line.split(' '):
                if index in index_to_country_state:
                    type = index_to_country_state[index][0]
                    value = index_to_country_state[index][1]
                    if type == "3": country = value
                    elif type == "4": state = value
            writer_name = '{}/{}'.format(country, state)
            output_folder = os.path.join(by_country_state_folder, writer_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                writers[writer_name] = open(os.path.join(output_folder, 'data'), 'w')
            writers[writer_name].write(line)

    # close all writers
    for writer in writers.values():
        writer.close()

def create_ngrams():
    names_dict = {}
    with open(output_names_dict_file) as reader:
        for line in reader:
            type, index, entity = line.strip().split('|')
            names_dict[index] = entity

    ngram_size = 3
    spaces = (' ' * ngram_size)
    for country in os.listdir(by_country_state_folder):
        country_folder = os.path.join(by_country_state_folder, country)
        states = ['.'] if 'data' in os.listdir(country_folder) else os.listdir(country_folder)
        for state in states:
            print 'working with', country, state
            country_state_folder = os.path.join(country_folder, state)

            # create ngrams
            ngrams = {}
            with open(os.path.join(country_state_folder, 'data')) as reader:
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
            with open(os.path.join(country_state_folder, 'ngrams'), 'w') as writer:
                for k1, v1 in ngrams.iteritems():
                    for k2, v2 in v1.iteritems():
                        writer.write('{}|{}|{}\n'.format(k1, k2, v2))

def create_names_symlinks():
    for country in os.listdir(by_country_state_folder):
        country_folder = os.path.join(by_country_state_folder, country)
        states = ['.'] if 'data' in os.listdir(country_folder) else os.listdir(country_folder)
        for state in states:
            country_state_folder = os.path.join(country_folder, state)
            output_link = os.path.join(country_state_folder, "names")
            if os.path.exists(output_link):
                os.remove(output_link)
            os.symlink(output_names_dict_file, output_link)

def create_common_ngrams_file():
    ngrams = {}
    for country in os.listdir(by_country_state_folder):
        country_folder = os.path.join(by_country_state_folder, country)
        states = ['.'] if 'data' in os.listdir(country_folder) else os.listdir(country_folder)
        for state in states:
            country_state_folder = os.path.join(country_folder, state)
            ngrams_file = os.path.join(country_state_folder, "ngrams")
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

def create_prefix_trees():
    names_dict = {}
    with open(output_names_dict_file) as reader:
        for line in reader:
            type, index, entity = line.strip().split('|')
            names_dict[index] = entity

    cpp_bindings.create_prefix_tree_builder()
    for country in os.listdir(by_country_state_folder):
        country_folder = os.path.join(by_country_state_folder, country)
        states = ['.'] if 'data' in os.listdir(country_folder) else os.listdir(country_folder)
        for state in states:
            print 'working with', country, state
            country_state_folder = os.path.join(country_folder, state)
            with open(os.path.join(country_state_folder, 'data')) as reader:
                for line in reader:
                    for index in line.strip().split(' '):
                        cpp_bindings.add_to_prefix_tree_builder(names_dict[index])
            cpp_bindings.finalize_prefix_tree_builder(os.path.join(country_state_folder, 'prefix-tree'))

if __name__ == '__main__':
    """
    print 'step 0: creating {} from {}'.format(input_file, output_folder)
    create_all_dataset_folder()
    """
    print 'step 1: separate {} to {}'.format(output_folder, by_country_state_folder)
    groups_dict = {"romania/data":"others",
                   "czech republic":"others",
                   "slovakia":"others",
                   "lithuania":"others",
                   "russia":"others",
                   "slovenia":"others",
                   "croatia":"others",
                   "latvia":"others",
                   "estonia":"others",
                   "luxembourg":"others",
                   "iceland":"others",
                   "greece":"others",
                   "ukraine":"others",
                   "bulgaria":"others",
                   "serbia":"others",
                   "cyprus":"others",
                   "san marino":"others",
                   "andorra":"others",
                   "liechtenstein":"others",
                   "montenegro":"others",
                   "monaco":"others",
                   "gibraltar":"others"}
    separate_by_country_state(groups_dict)
    print 'step 2: creating ngrams'
    create_ngrams()
    print 'step 3: creating symlinks for names'
    create_names_symlinks()
    print 'step 4: creating common ngrams file'
    create_common_ngrams_file()
    print 'step 5: create prefix tree'
    create_prefix_trees()


