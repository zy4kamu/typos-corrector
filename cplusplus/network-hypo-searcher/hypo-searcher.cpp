#include "hypo-searcher.h"

#include <boost/algorithm/string/join.hpp>

#include <fstream>
#include <sstream>

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

class GridIterator {
public:
    GridIterator(std::vector<size_t>&& _sizes): sizes(std::move(_sizes)), current_indexes(sizes.size(), 0), size(1) {
        for (const size_t dimension_size : sizes) {
            size *= dimension_size;
        }
    }

    const std::vector<size_t>& get() const {
        return current_indexes;
    }

    void next() {
        for (size_t i = 0; i < sizes.size(); ++i) {
            current_indexes[i] += 1;
            if (current_indexes[i] == sizes[i]) {
                current_indexes[i] = 0;
            } else {
                break;
            }
        }
    }

    size_t get_size() const {
        return size;
    }
private:
    std::vector<size_t> sizes;
    std::vector<size_t> current_indexes;
    size_t size;
};

void take_smallest_strings_only(std::vector<std::string>& tokens) {
    size_t size = std::numeric_limits<size_t>::max();
    for (const std::string& token : tokens) {
        if (token.length() < size) {
            size = token.length();
        }
    }
    auto iter = std::remove_if(tokens.begin(), tokens.end(), [size](const std::string& token) {
        return token.length() > size; });
    tokens.resize(std::distance(tokens.begin(), iter));
}

const size_t MAX_HYPOS = 20;
const size_t MAX_NUMBER_OF_ATTEMPTS = 100;

} // anonymous namespace

bool HypoNodePointerComparator::operator()(const HypoNode* first, const HypoNode* second) const {
    assert(first != nullptr);
    assert(second != nullptr);
    return (first->logit > second->logit) || ((first->logit == second->logit) && (first > second));
}

HypoSearcher::HypoSearcher(const std::string& dataset_folder,
                           const std::string& lstm_folder,
                           const std::string& first_mistake_file)
    : automata(lstm_folder), dataset(dataset_folder) {
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
            for (size_t j = 0; j < EFFECTIVE_NUM_LETTERS; ++j) {
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
                for (size_t j = 0; j < EFFECTIVE_NUM_LETTERS; ++j) {
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
        std::vector<std::vector<std::string>> hypos = find_max_prefix_several_tokens(hypo, max_prefix_length);

        // Create grid iterator
        std::vector<size_t> sizes(hypos.size());
        for (size_t i = 0; i < hypos.size(); ++i) {
            sizes[i] = hypos[i].size();
        }
        GridIterator iterator(std::move(sizes));

        // If this is a perfect match return result
        if (max_prefix_length == hypo.length()) {
            std::vector<std::string> perfect_hypos;
            for (size_t i = 0; i < iterator.get_size(); ++i) {
                const std::vector<size_t>& indexes = iterator.get();
                std::string concatenated;
                for (size_t j = 0; j < indexes.size(); ++j) {
                    if (!concatenated.empty()) {
                        concatenated += ' ';
                    }
                    concatenated += hypos[j][indexes[j]];
                }
                perfect_hypos.push_back(std::move(concatenated));
            }
            return perfect_hypos;
        }

        // check if one of hypos is acceptable by levenstein distance
        size_t best_levenstein_distance = 4;
        std::vector<std::string> levenstein_hypos;
        for (size_t i = 0; i < iterator.get_size(); ++i) {
            const std::vector<size_t>& indexes = iterator.get();
            std::string concatenated;
            for (size_t j = 0; j < indexes.size(); ++j) {
                if (!concatenated.empty()) {
                    concatenated += ' ';
                }
                concatenated += hypos[j][indexes[j]];
            }
            size_t distance = levenstein_distance(input_token, concatenated);
            if (distance < best_levenstein_distance) {
                levenstein_hypos.clear();
                levenstein_hypos.push_back(std::move(concatenated));
            } else if (distance == best_levenstein_distance) {
                levenstein_hypos.push_back(std::move(concatenated));
            }
            iterator.next();
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

std::vector<std::vector<std::string>>
HypoSearcher::find_max_prefix_several_tokens(const std::string& string, size_t& max_prefix_length) const {
    std::stringstream reader(string);
    std::vector<std::vector<std::string>> resulted_hypos;
    std::string current_token;
    while (getline(reader, current_token, '|')) {
        if (contains_digit(current_token)) {
            resulted_hypos.push_back({ current_token });
            max_prefix_length += max_prefix_length > 0 ? current_token.length() + 1 : current_token.length();
        } else if (reader.peek() == EOF) {
            size_t one_token_max_prefix_length;
            resulted_hypos.push_back(find_max_prefix_one_token(current_token, one_token_max_prefix_length));
            take_smallest_strings_only(resulted_hypos.back());
            max_prefix_length += max_prefix_length > 0 ? one_token_max_prefix_length + 1 : one_token_max_prefix_length;
        }
        else {
            size_t one_token_max_prefix_length;
            std::vector<std::string> hypos = find_max_prefix_one_token(current_token, one_token_max_prefix_length);
            take_smallest_strings_only(hypos);
            max_prefix_length += max_prefix_length > 0 ? current_token.length() + 1 : one_token_max_prefix_length;
            if (hypos.empty()) {
                break;
            }
            resulted_hypos.push_back(std::move(hypos));
        }
    }
    return resulted_hypos;
}

std::vector<std::string> HypoSearcher::find_max_prefix_one_token(const std::string& token, size_t& max_prefix_length) const {
    std::vector<std::string> hypos = dataset.find_by_prefix(token, MAX_HYPOS);
    if (!hypos.empty()) {
        max_prefix_length = token.length();
        return hypos;
    }
    size_t start = 0;
    size_t end = token.length();
    while (start + 1 < end) {
        size_t middle = (start + end) / 2;
        hypos = dataset.find_by_prefix(token.substr(0, middle), MAX_HYPOS);
        if (hypos.empty()) {
            end = middle;
        } else {
            start = middle;
        }
    }
    max_prefix_length = start;
    return hypos;
}

void HypoSearcher::read_first_mistake_statistics(const std::string& first_mistake_file) {
    std::ifstream reader(first_mistake_file);
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
