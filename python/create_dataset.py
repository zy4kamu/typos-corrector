import os
import unicodedata

import utils

input_file = '/home/stepan/datasets/europe-hierarchy/preprocessed'
output_names_dict_file = '/home/stepan/git-repos/typos-corrector/python/model/dataset/names'
output_transitions_file = '/home/stepan/git-repos/typos-corrector/python/model/dataset/transitions'

#### Functions to create names_dict and transitions_dict

def strip_accents(input):
    return unicodedata.normalize('NFKD', input.decode('utf-8')).encode('ASCII', 'ignore')

names_dict = {}
transitions_dict = {}

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

    # !!! REMOVE IT LATER
    ok = 'netherlands' in buffer[0] or 'germany' in buffer[0] or 'italy' in buffer[0]
    if not ok: return

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

def work():
    buffer = []
    counter = 0
    with open(input_file) as reader:
        for line in reader:
            line = strip_accents(line.strip().lower())
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

if __name__ == '__main__':
    if os.path.exists(output_transitions_file): os.remove(output_transitions_file)
    if os.path.exists(output_names_dict_file): os.remove(output_names_dict_file)
    work()
    print 'created 2 files:', output_names_dict_file, 'and', output_transitions_file
    print 'now sort and remove duplicates from', output_transitions_file