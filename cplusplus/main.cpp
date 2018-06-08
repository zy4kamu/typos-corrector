#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/make_unique.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "contaminator.h"
#include "dictionary.h"
#include "prefix-tree.h"
#include "prefix-tree-builder.h"
#include "random-batch-generator.h"
#include "utils.h"

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    std::string command;
    std::string text;
    size_t min_chars;
    size_t max_chars;
    std::string prefix_tree_file;
    std::string dictionary_file;

    po::options_description desc("Prepares batches for typos correction train and test");
    try {
        desc.add_options()("command,c", po::value<std::string>(&command)->required(), 
                           "task to do: prefix-tree");
        desc.add_options()("prefix-tree-file,p", po::value<std::string>(&prefix_tree_file)->default_value("prefix-tree"),
                           "output file with binary prefix tree");
        desc.add_options()("dictionary-file,d", po::value<std::string>(&dictionary_file)->default_value("dictionary-file"),
                           "output file with dictionary");
        desc.add_options()("min-chars,mn", po::value<size_t>(&min_chars)->default_value(3),
                           "min number of characters in token in order to take it into account");
        desc.add_options()("max-chars,mx", po::value<size_t>(&max_chars)->default_value(30),
                           "max number of characters in token in order to take it into account");
        po::variables_map variables_map;
        po::store(po::parse_command_line(argc, argv, desc), variables_map);
        po::notify(variables_map);    
    } catch (std::exception&) {
        std::cout << desc << std::endl;
        exit(0);
    }

    if (command == "prefix-tree") {
        PrefixTreeBuilder builder(dictionary_file, prefix_tree_file, min_chars, max_chars);
        builder.build_prefix_tree();
    } else {
        std::cerr << "Invalid command parameter" << std::endl;
        std::cout << desc << std::endl;
    }
}


