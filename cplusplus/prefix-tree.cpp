#include "prefix-tree.h"

#include <boost/make_unique.hpp>
#include <cstring>
#include <fstream>
#include <string>

#include "utils.h"

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

PrefixTree::PrefixTree(): root(boost::make_unique<Node>())
                        , generator(1) {
}

PrefixTree::PrefixTree(const std::string& input_file): generator(1) {
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

std::string PrefixTree::generate() const {
    const PrefixTree::Node* node = root.get();
    std::string token;
    char current_char = '\0';
    while (true) {
        size_t num_transitions = node->transitions.size();
        std::uniform_int_distribution<size_t> distribution(0, num_transitions - 1);
        auto iter = node->transitions.begin();
        std::advance(iter, distribution(generator));
        current_char = iter->first;
        node = iter->second.get();
        if (current_char != '$') {
            token += current_char;
        } else {
            break;
        }
    }
    return token;
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

/*
std::vector<std::pair<float, std::string>> PrefixTree::viterbi(const std::vector<std::vector<float>>& probailities,
                                                               size_t num_hypos) const {
    size_t grid_length = probailities.size();
    size_t num_choices = probailities[0].size();

    // initialize first layer
    std::vector<std::multimap<float, PrefixTree::Node*>> grid(num_choices);
    for (size_t i = 0; i < num_choices; ++i) {
        auto found = root->transitions.find(to_char(i));
        if (found != root->transitions.end()) {
            grid[i] = { probailities[0][i], found->second.get() };
        }
    }

    // run over grid
    for (size_t time = 1; time < grid_length; ++time) {
        std::vector<std::multimap<float, PrefixTree::Node*>> updated_grid(num_choices);
        for (const std::multimap<float, PrefixTree::Node*>& item : grid) {
            for (const auto& kvp : item) {
                float current_probability = kvp.first;
                const std::unordered_map<char, std::unique_ptr<Node>>& transitions = item.second->transitions;
                for (const auto& kvp2 : transitions) {
                    char next_char = kvp2.first;
                    PrefixTree::Node* next_node = kvp2.second.get();
                    updated_grid[next_char] = { current_probability + probailities[time][to_int(next_char)], next_node };
                }
                updated_grid[next_char] = { current_probability + probailities[time][to_int(' ')], item.second.get() };
            }
        }
        for (std::map<float, PrefixTree::Node*>& item : updated_grid) {
            while (item.size() > num_hypos) {
                item.erase(item.begin());
            }
        }
        grid = std::move(updated_grid);
    }

    // fill result nodes
    std::multimap<float, PrefixTree::Node*> node_results;
    for (const std::multimap<float, PrefixTree::Node*>& item : grid) {
        node_results.insert(item);
    }
    while (node_results.size() > num_hypos) {
        node_results.erase(node_results.begin());
    }

    std::vector<std::pair<float, std::string>>
}
*/

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

void generate(char* token) {
    std::string generated = prefix_tree->generate();
    std::memcpy(token, generated.c_str(), generated.size());
}

} // extern "C"
