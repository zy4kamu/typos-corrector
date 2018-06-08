#include "prefix-tree-builder.h"
#include "prefix-tree.h"

#include <boost/make_unique.hpp>
#include <iostream>
#include <memory>

PrefixTreeBuilder::PrefixTreeBuilder(const std::string& input_file, const std::string& output_prefix_tree_file,
                                     size_t min_chars, size_t max_chars)
    : output_prefix_tree_file(output_prefix_tree_file)
    , dictionary(input_file, min_chars, max_chars) {
}

void PrefixTreeBuilder::build_prefix_tree() {
    std::unique_ptr<PrefixTree> prefix_tree = boost::make_unique<PrefixTree>();
    for (const std::string& token : dictionary.get()) {
        prefix_tree->add(token);
    }
    std::cout << "saving tree to " << output_prefix_tree_file << std::endl;
    prefix_tree->save(output_prefix_tree_file);
}
