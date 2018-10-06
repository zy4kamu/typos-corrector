#include "hypo-searcher.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <fstream>
#include <sstream>

#include "utils.h"

namespace NNetworkHypoSearcher {

namespace {

size_t argmax(const std::vector<float_type>& data) {
    float_type max = std::numeric_limits<float_type>::min();
    size_t best_index = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] > max) {
            max = data[i];
            best_index = i;
        }
    }
    return best_index;
}

void recursive_erase(HypoNode* node_to_erase, AutomataNodesSet& nodes_to_process) {
    nodes_to_process.erase(node_to_erase);
    for (HypoNode& node : node_to_erase->transitions) {
        recursive_erase(&node, nodes_to_process);
    }
}

} // anonymous namespace

bool HypoNodePointerComparator::operator()(const HypoNode* first, const HypoNode* second) const {
    assert(first != nullptr);
    assert(second != nullptr);
    return (first->logit > second->logit) || ((first->logit == second->logit) && (first > second));
}

HypoSearcher::HypoSearcher(const std::string& lstm_folder)
    : lstm_folder(lstm_folder)
    , automata(lstm_folder)
    , max_prefix_length(std::string::npos)
    , current_probabilities(NUM_LETTERS) {
    read_first_mistake_statistics(lstm_folder + "/first-mistake-statistics");
}

void HypoSearcher::load() {
    automata.load();
}

void HypoSearcher::unload() {
    automata.unload();
}

bool HypoSearcher::is_loaded() const {
    return automata.is_loaded();
}

void HypoSearcher::reset() {
    root = { '*', first_mistake_statistics[0], "" };
    nodes_to_process.clear();
    nodes_to_process.insert(&root);
}

void HypoSearcher::initialize(const std::string& input) {
    initial_input = input.substr(0, std::min(input.length(), MESSAGE_SIZE));
    max_prefix_length = std::string::npos;
    reset();
    automata.encode_message(input, current_probabilities);
}

const std::string& HypoSearcher::generate_next_hypo() {
    // Delete nodes which deninitely will not appear in final hypo
    if (max_prefix_length != std::string::npos) {
        HypoNode* node_to_delete = &root;
        for (size_t i = 0; i < max_prefix_length + 1; ++i) {
            node_to_delete = &node_to_delete->transitions[to_int(current_hypo[i])];
        }
        recursive_erase(node_to_delete, nodes_to_process);
    }
    max_prefix_length = std::string::npos;

    // Take the best hypo and get to the state from where we can start searching hypos
    automata.reset_pass();
    HypoNode* current_node = *nodes_to_process.begin();
    for (const char letter : current_node->prefix) {
        automata.apply(letter, current_probabilities);
    }
    current_hypo = current_node->prefix;
    nodes_to_process.erase(nodes_to_process.begin());

    for (size_t i = current_hypo.length(); i < MESSAGE_SIZE; ++i) {
        // update automata nodes
        for (size_t j = 0; j < EFFECTIVE_NUM_LETTERS; ++j) {
            char letter_to_add = to_char(static_cast<int32_t>(j));
            float_type logit = current_node->logit + std::log(current_probabilities[j]) + first_mistake_statistics[i + 1] -
                    first_mistake_statistics[i];
            current_node->transitions.emplace_back(letter_to_add, logit, current_hypo + letter_to_add);
        }

        // choose next best char
        if (i + 1 < MESSAGE_SIZE) {
            size_t index = argmax(current_probabilities);
            char ch = to_char(static_cast<int32_t>(index));
            current_hypo += ch;
            for (size_t j = 0; j < EFFECTIVE_NUM_LETTERS; ++j) {
                if (j != index) {
                    nodes_to_process.insert(&current_node->transitions[j]);
                }
            }
            current_node = &current_node->transitions[index];
            automata.apply(ch, current_probabilities);
        }
    }

    // check if hypo is present in the dictionary
    current_hypo.erase(current_hypo.find_last_not_of(' ') + 1);

    // calculate current levenstein between hypo and initial request
    std::string temp_hypo = current_hypo;
    boost::replace_all(temp_hypo, "|", " ");
    current_levenstein = levenstein_distance(initial_input, temp_hypo);

    return current_hypo;
}

bool HypoSearcher::check_hypo_in_database(IDataBaseRequester& requester, std::string& levenstein_correction) {
    size_t prefix_length = requester.levenstein_request(current_hypo, 10,
                                                        static_cast<size_t>(std::max(0, 4 - static_cast<int>(current_levenstein))),
                                                        '|', levenstein_correction);
    if (max_prefix_length == std::string::npos || prefix_length > max_prefix_length) {
        max_prefix_length = prefix_length;
    }
    return max_prefix_length == current_hypo.length() || !levenstein_correction.empty();
}

void HypoSearcher::read_first_mistake_statistics(const std::string& first_mistake_file) {
    std::ifstream reader(first_mistake_file);
    std::string line;
    float_type unnormalized_sum = 0;
    while (getline(reader, line)) {
        first_mistake_statistics.push_back(static_cast<float_type>(std::stof(line) + 1.0));
        unnormalized_sum += first_mistake_statistics.back();
    }
    probability_not_to_correct = 1.0 - first_mistake_statistics.back() / unnormalized_sum;
    for (size_t i = first_mistake_statistics.size() - 1; i + 1 != 0; --i) {
        first_mistake_statistics[i] += first_mistake_statistics[i + 1];
    }
    for (float_type& item : first_mistake_statistics) {
        item = std::log(first_mistake_statistics.back() / item);
    }
    float_type sum = 0;
    for (float_type item : first_mistake_statistics) {
        sum += item;
    }
}

float_type HypoSearcher::get_probability_not_to_correct() const {
    return probability_not_to_correct;
}

} // namespace NNetworkHypoSearcher
