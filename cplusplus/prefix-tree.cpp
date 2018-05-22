#include "prefix-tree.h"

#include <boost/make_unique.hpp>
#include <cstring>
#include <fstream>
#include <string>

/***** Helper functions *****/

namespace {

template <class T>
void read_binary(std::ifstream& reader, T& item) {
    reader.read((char*)&item, sizeof(T));
}

template <class T>
void write_binary(std::ofstream& writer, const T& item) {
    writer.write((char*)&item, sizeof(T));
}

} // anonymous namespace

/***** Implementation *****/

PrefixTree::PrefixTree(): root(boost::make_unique<Node>()) {
}

PrefixTree::PrefixTree(const std::string& input_file) {
    std::ifstream reader(input_file, std::ios::binary);
    root = read_node(reader);
}

size_t PrefixTree::match(const std::string& token) const {
    return match(token.data(), token.size());
}

size_t PrefixTree::match(const char* token) const {
    return strlen(token);
}

size_t PrefixTree::match(const char* token, size_t length) const {
    const PrefixTree::Node* node = root.get();
    for (size_t i = 0; i < length; ++i) {
        auto found = node->transitions.find(token[i]);
        if (found == node->transitions.end()) {
            return i;
        }
        node = found->second.get();
    }
    return length;
}

void PrefixTree::add(const std::string& token) {
    add(token.c_str(), token.length());
}

void PrefixTree::add(const char* token, size_t length) {
    Node* node = root.get();
    for (size_t i = 0; i < length + 1; ++i) {
        char letter = i < length ? token[i] : '$';
        std::unique_ptr<Node>& node_ptr = node->transitions[letter];
        if (!node_ptr) {
            node_ptr = boost::make_unique<PrefixTree::Node>();
        }
        node = node_ptr.get();
    }
}

void PrefixTree::save(const std::string& output_file) const {
    std::ofstream writer(output_file, std::ios::binary);
    write_node(writer, *root);
}

std::unique_ptr<PrefixTree::Node> PrefixTree::read_node(std::ifstream& reader) const {
     std::unique_ptr<PrefixTree::Node> node = boost::make_unique<PrefixTree::Node>();
     size_t num_transitions = 0;
     read_binary(reader, num_transitions);
     for (size_t i = 0; i < num_transitions; ++i) {
         char letter;
         read_binary(reader, letter);
         node->transitions[letter] = read_node(reader);
     }
     return node;
}

void PrefixTree::write_node(std::ofstream& writer, const PrefixTree::Node& node) const {
    write_binary(writer, node.transitions.size());
    for (const auto& item : node.transitions) {
        write_binary(writer, item.first);
        write_node(writer, *item.second);
    }
}

/********* Python bidings *********/

extern "C" {

std::unique_ptr<PrefixTree> prefix_tree;

void create() {
    prefix_tree = boost::make_unique<PrefixTree>();
}

void create_from_file(const char* file_name, size_t size) {
    prefix_tree = boost::make_unique<PrefixTree>(std::string(file_name, size));
}

void destroy() {
    prefix_tree.release();
}

size_t match(const char* token, size_t length) {
    return prefix_tree->match(token, length);
}

void add(const char* token, size_t length) {
    prefix_tree->add(token, length);
}

} // extern "C"
