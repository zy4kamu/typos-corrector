#include "hypo-searcher.h"

#include <iostream>
#include <fstream>

#include "../utils/utils.h"

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

const size_t MAX_HYPOS = 20;
const size_t MAX_NUMBER_OF_ATTEMPTS = 100;

} // anonymous namespace

bool HypoNodePointerComparator::operator()(const HypoNode* first, const HypoNode* second) {
    assert(first != nullptr);
    assert(second != nullptr);
    return (first->logit > second->logit) || ((first->logit == second->logit) && (first > second));
}

HypoSearcher::HypoSearcher(const boost::filesystem::path& update_regions_folder,
                           const boost::filesystem::path& lstm_folder,
                           const boost::filesystem::path& first_mistake_file)
    : automata(lstm_folder), update_regions(update_regions_folder), compressor(update_regions, MESSAGE_SIZE) {
    read_first_mistake_statistics(first_mistake_file);
}

void HypoSearcher::reset() {
    root = { '*', first_mistake_statistics[0], "" };
    nodes_to_process.clear();
    nodes_to_process.insert(&root);
}

std::vector<std::string> HypoSearcher::search(const std::string& input_token) {
    // encode message and prepare automata
    reset();
    std::vector<float_type> probabilities(NUM_LETTERS);
    automata.encode_message(input_token, probabilities);

    size_t counter = 0;
    while (!nodes_to_process.empty() && ++counter < MAX_NUMBER_OF_ATTEMPTS) {
        // Take the best hypo and get to the state from where we can start searching hypos
        automata.reset();
        HypoNode* current_node = *nodes_to_process.begin();
        for (const char letter : current_node->prefix) {
            automata.apply(letter, probabilities);
        }
        std::string hypo = current_node->prefix;
        nodes_to_process.erase(nodes_to_process.begin());

        for (size_t i = hypo.length(); i < MESSAGE_SIZE; ++i) {
            // update automata nodes
            for (size_t j = 0; j < NUM_LETTERS; ++j) {
                char letter_to_add = to_char(static_cast<int32_t>(j));
                float_type logit = current_node->logit + std::log(probabilities[j]) + first_mistake_statistics[i + 1] -
                        first_mistake_statistics[i];
                current_node->transitions.emplace_back(letter_to_add, logit, hypo + letter_to_add);
            }

            // choose next best char
            if (i + 1 < MESSAGE_SIZE) {
                size_t index = argmax(probabilities);
                char ch = to_char(static_cast<int32_t>(index));
                hypo += ch;
                for (size_t j = 0; j < NUM_LETTERS; ++j) {
                    if (j != index) {
                        nodes_to_process.insert(&current_node->transitions[j]);
                    }
                }
                current_node = &current_node->transitions[index];
                automata.apply(ch, probabilities);
            }
        }

        // check if hypo is present in the dictionary
        hypo.erase(hypo.find_last_not_of(' ') + 1);
        size_t max_prefix_length = 0;
        std::vector<std::string> hypos = find_max_prefix(hypo, max_prefix_length);
        if (max_prefix_length == hypo.length()) {
            return hypos;
        }

        // check if one of hypos is acceptable by levenstein distance
        size_t best_levenstein_distance = 4;
        std::vector<std::string> levenstein_hypos;
        for (std::string& found_hypo : hypos) {
            size_t distance = levenstein_distance(input_token, found_hypo);
            if (distance < best_levenstein_distance) {
                levenstein_hypos.clear();
                levenstein_hypos.push_back(std::move(found_hypo));
            } else if (distance == best_levenstein_distance) {
                levenstein_hypos.push_back(found_hypo);
            }
        }
        if (!levenstein_hypos.empty()) {
            return levenstein_hypos;
        }

        // erase everything after max_prefix_length
        HypoNode* node_to_delete = &root;
        for (size_t i = 0; i < max_prefix_length + 1; ++i) {
            node_to_delete = &node_to_delete->transitions[to_int(hypo[i])];
        }
        recursive_erase(node_to_delete, nodes_to_process);
    }

    return {};
}

std::vector<std::string> HypoSearcher::find_max_prefix(const std::string& token, size_t& max_prefix_length) const {
    std::vector<std::string> hypos = compressor.find_by_prefix(token, MAX_HYPOS);
    if (!hypos.empty()) {
        max_prefix_length = token.length();
        return hypos;
    }
    size_t start = 0;
    size_t end = token.length();
    while (start + 1 < end) {
        size_t middle = (start + end) / 2;
        hypos = compressor.find_by_prefix(token.substr(0, middle), MAX_HYPOS);
        if (hypos.empty()) {
            end = middle;
        } else {
            start = middle;
        }
    }
    max_prefix_length = start;
    return hypos;
}

void HypoSearcher::read_first_mistake_statistics(const boost::filesystem::path& first_mistake_file) {
    std::ifstream reader(first_mistake_file.string());
    std::string line;
    while (getline(reader, line)) {
        first_mistake_statistics.push_back(static_cast<float_type>(std::stof(line) + 1.0));
    }
    for (size_t i = first_mistake_statistics.size() - 1; i + 1 != 0; --i) {
        first_mistake_statistics[i] += first_mistake_statistics[i + 1];
    }
    for (float_type& item : first_mistake_statistics) {
        item = std::log(first_mistake_statistics.back() / item);
    }
}

} // namespace NNetworkHypoSearcher
