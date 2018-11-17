#pragma once

#include <string>

#include "common.h"
#include "network-automata-cpu.h"
#include "../prefix-tree/prefix-tree.h"

using NetworkAutomata = NNetworkHypoSearcher::NetworkAutomataCPU;

namespace NNetworkHypoSearcher {

struct PrefixTreeHypoSearcherNode : public PrefixTreeNode {
    PrefixTreeHypoSearcherNode(const char* root, float_type probability, std::string&& prefix, std::vector<float_type>&& transition_probabilities):
      PrefixTreeNode(root), probability(probability), prefix(std::move(prefix)), transition_probabilities(std::move(transition_probabilities)) {
    }

    float_type probability;
    std::string prefix;
    std::vector<float_type> transition_probabilities;

    void walk(NetworkAutomata& automata, float_type target_probability,
              float_type& covered_probability, std::vector<std::string>& hypos);
};

class PrefixTreeHypoSearcher {
public:
    PrefixTreeHypoSearcher(const std::string& lstm_folder, const std::string& prefix_tree_file);
    std::vector<std::string> process(const std::string& input, float_type target_probability);
    void load() { automata.load(); }
private:
    std::string lstm_folder;
    NetworkAutomata automata;
    std::string initial_input;
    std::vector<float_type> initial_probabilities;
    PrefixTree prefix_tree;
};

} // namespace NNetworkHypoSearcher
