#include "prefix-tree.h"
#include "utils.h"

#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
    const std::string input_file = argv[1];
    const std::string output_file = argv[2];
    PrefixTree::create(input_file, output_file);
}
