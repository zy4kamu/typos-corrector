"""
Given SQL dump creates a folder with hierarchic structure: country -> city -> street
"""

import argparse, os, shutil
import utils


def create_dataset(input_file, output_folder):
    def line_contain_address(line):
        """ Checks whether line from SQL dump contains address """
        splitted = line.split(',')
        return len(splitted) > 6 and splitted[6] == ' 2' and \
               line.startswith('INSERT INTO poiSecondaryAttributes(poiId, grp, typeId, typeFormat, val) VALUES')

    def clean_string(input):
        """ Removes non-alphabetical characters + to lower + remove diacritics """
        input = utils.strip_accents(input)
        input = input.strip().lower()
        return ''.join([_ for _ in input if utils.acceptable(_)])

    # clean output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # parse SQL dump and parse city_to_street
    city_to_street = {}
    with open(input_file) as reader:
        while True:
            line = reader.readline()
            if not line: break
            if line_contain_address(line):
                next_line = reader.readline() # address is separated by '\n'; therefore we have to append next line
                line = line.strip() + next_line
                splitted_address = ''.join(line.split("'")[1:]).split('\x1c')
                if len(splitted_address) > 1:
                    street = clean_string(splitted_address[0])
                    city = clean_string(splitted_address[-2])
                    if len(street) == 0 or len(city) == 0: continue
                    if not city in city_to_street:
                        city_to_street[city] = set()
                    city_to_street[city].add(street)

    # TODO: support all countries
    country = 'netherlands'
    os.makedirs(os.path.join(output_folder, country))
    for city, streets in city_to_street.iteritems():
        with open(os.path.join(output_folder, country + '/' + city), 'w') as writer:
            writer.write('\n'.join(sorted(list(streets))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test neural network for typos correction')
    parser.add_argument('-i', '--input-file', type=str, help='input sql dump folder', required=True)
    parser.add_argument('-o', '--output-folder', type=str, help='output folder with countries/cities/streets', required=True)
    args = parser.parse_args()

    create_dataset(args.input_file, args.output_folder)
