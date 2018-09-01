#pragma once

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "network-automata.h"
#include "../dataset/compressor.h"

namespace NNetworkHypoSearcher {

struct HypoNode {
    HypoNode() = default;
    HypoNode(char letter, float_type logit, std::string&& prefix)
        : letter(letter), logit(logit), prefix(std::move(prefix)) {
    }

    char letter;
    float_type logit;
    std::string prefix; // TODO: replace with NN state
    std::vector<HypoNode> transitions;
};

struct HypoNodePointerComparator {
    bool operator()(const HypoNode* first, const HypoNode* second) const;
};

using AutomataNodesSet = std::set<HypoNode*, HypoNodePointerComparator>;

class HypoSearcher {
public:
    HypoSearcher(const std::string& dataset_folder,
                 const std::string& lstm_folder,
                 const std::string& first_mistake_file);
    std::vector<std::string> search(const std::string& input_token);
private:
    void read_first_mistake_statistics(const std::string& first_mistake_file);
    void reset();
    std::vector<std::vector<std::string>> find_max_prefix_several_tokens(const std::string& string,
                                                                         size_t& max_prefix_length) const;
    std::vector<std::string> find_max_prefix_one_token(const std::string& token, size_t& max_prefix_length) const;

    NetworkAutomata         automata;
    std::vector<float_type> first_mistake_statistics;
    DataSet                 dataset;
    Compressor              compressor;

    AutomataNodesSet nodes_to_process;
    HypoNode root;
};

} // namespace NNetworkHypoSearcher
