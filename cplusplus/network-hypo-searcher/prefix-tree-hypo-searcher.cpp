#include "prefix-tree-hypo-searcher.h"
#include "utils.h"

#include <algorithm>

namespace NNetworkHypoSearcher {

void PrefixTreeHypoSearcherNode::walk(NetworkAutomata& automata, float_type target_probability,
                                  float_type& covered_probability, std::vector<std::string>& hypos) {
    // reach end means to find a hypo
    if (transitions.empty()) {
        covered_probability += probability;
        hypos.push_back(prefix);
    }

    if (covered_probability > target_probability) {
        return;
    }

    // get all possible transitions
    std::vector<size_t> transition_indexes;
    for (uint32_t i = 0; i < EFFECTIVE_NUM_LETTERS; ++i) {
        char letter = to_char(i);
        if (std::find(transitions.begin(), transitions.end(), letter) != transitions.end()) {
            transition_indexes.push_back(i);
        } else {
            covered_probability += probability * transition_probabilities[i];
        }
    }
    if (covered_probability > target_probability) {
        return;
    }
    std::sort(transition_indexes.begin(), transition_indexes.end(),
              [this](size_t i, size_t j) { return transition_probabilities[i] > transition_probabilities[j]; });

    // go deep
    for (size_t transition_index : transition_indexes) {
        char letter = to_int(transition_index);
        PrefixTreeNode node = move(letter);
        std::string next_prefix = prefix + letter;
        std::vector<float_type> next_transition_probabilities(EFFECTIVE_NUM_LETTERS);
        automata.reset_pass();
        for (const char letter : next_prefix) {
            automata.apply(letter, next_transition_probabilities);
        }
        PrefixTreeHypoSearcherNode next_node(node.data,
                                             probability * transition_probabilities[transition_index],
                                             std::move(next_prefix),
                                             std::move(next_transition_probabilities));
        next_node.walk(automata, target_probability, covered_probability, hypos);
        if (covered_probability > target_probability) {
            return;
        }
    }
}

PrefixTreeHypoSearcher::PrefixTreeHypoSearcher(const std::string& lstm_folder, const std::string& prefix_tree_file)
    : lstm_folder(lstm_folder)
    , automata(lstm_folder)
    , initial_probabilities(EFFECTIVE_NUM_LETTERS)
    , prefix_tree(prefix_tree_file) {
}

std::vector<std::string> PrefixTreeHypoSearcher::process(const std::string& input, float_type target_probability) {
    initial_input = input.substr(0, std::min(input.length(), MESSAGE_SIZE));
    automata.encode_message(input, initial_probabilities);

    PrefixTreeHypoSearcherNode root(prefix_tree.get_root(), 1, "", std::move(initial_probabilities));
    float_type covered_probability = 0;
    std::vector<std::string> hypos;
    root.walk(automata, target_probability, covered_probability, hypos);
    return hypos;
}

} // namespace NNetworkHypoSearcher
