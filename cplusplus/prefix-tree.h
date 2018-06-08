#pragma once

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

class PrefixTree {
public:
    PrefixTree();
    PrefixTree(const std::string& file_name);
    size_t match(const std::string& token) const;
    size_t match(const char* token) const;
    size_t match(const char* token, size_t length) const;
    void add(const std::string& token);
    void add(const char* token, size_t length);
    void save(const std::string& file_name) const;
    std::string generate() const;
    /*
    std::vector<std::pair<float, std::string>> viterbi(const std::vector<std::vector<float>>& probailities,
                                                       size_t num_hypos) const;
                                                       */
private:
    struct Node {
        std::unordered_map<char, std::unique_ptr<Node>> transitions;
    };

    void write_node(std::ofstream& writer, const Node& node) const;
    std::unique_ptr<Node> read_node(std::ifstream& reader) const;

    std::unique_ptr<Node> root;
    mutable std::mt19937 generator;
};
