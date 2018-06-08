#pragma once

#include <algorithm>
#include <string>

#include "dictionary.h"

class PrefixTreeBuilder {
public:
    PrefixTreeBuilder(const std::string& input_file, const std::string& output_prefix_tree_file,
                    size_t min_chars, size_t max_chars);
    void build_prefix_tree();
private:
    std::string output_prefix_tree_file;
    Dictionary dictionary;
};



