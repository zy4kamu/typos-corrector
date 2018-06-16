#include "prefix-tree.h"
#include "utils.h"

#include <cassert>
#include <cstring>
#include <boost/make_unique.hpp>
#include <fstream>
#include <iostream>
#include <set>
#include <unordered_map>

namespace {

class DecompressedPrefixTree {
private:
    struct Node {
        std::unordered_map<char, std::unique_ptr<Node>> transitions;

        String to_string() const {
            if (transitions.empty()) {
                return String(1, 0);
            } else if (transitions.size() == 1) {
                unsigned char letter = static_cast<unsigned char>(transitions.begin()->first);
                return letter + transitions.begin()->second->to_string();
            } else {
                std::vector<unsigned char> chars;
                std::vector<String> subtrees;
                for (const auto& item : transitions) {
                    chars.push_back(item.first);
                    subtrees.push_back(item.second->to_string());
                }
                String result;
                uint32_t offset = 4 * chars.size() + 1;
                for (size_t i = 0; i < chars.size(); ++i) {
                    result.push_back(128 + chars[i]);
                    result.push_back(offset / (256 * 256));
                    result.push_back((offset % (256 * 256)) / 256);
                    result.push_back(offset % 256 );
                    offset += subtrees[i].size() - 4;
                }
                result.push_back(0);
                for (size_t i = 0; i < chars.size(); ++i) {
                    result += subtrees[i];
                }
                return result;
            }
        }
    };
public:
    DecompressedPrefixTree(): root(boost::make_unique<Node>()) {
    }

    void add(const std::string& token) {
        Node* node = root.get();
        for (size_t i = 0; i < token.length(); ++i) {
            char letter = token[i];
            std::unique_ptr<Node>& node_ptr = node->transitions[letter];
            if (!node_ptr) {
                node_ptr = boost::make_unique<Node>();
            }
            node = node_ptr.get();
        }
    }

    String to_string() const {
        return root->to_string();
    }

    std::unique_ptr<Node> root;
};

size_t get_file_size(const std::string& input_file) {
    std::ifstream reader(input_file, std::ifstream::ate | std::ifstream::binary);
    return reader.tellg();
}

struct ViterbiState {
    std::string prefix;
    size_t offset;
    double logit;
    bool terminated = false;

    bool operator <(const ViterbiState& other) const {
        return (logit < other.logit) || (logit == other.logit && offset < other.offset);
    }

    bool operator ==(const ViterbiState& other) const {
        return offset == other.offset;
    }
};

bool pre_insert(size_t step, std::set<ViterbiState>& states, double logit, size_t num_hypos) {
    if (states.size() < num_hypos) {
        return true;
    } else if (logit < states.begin()->logit) {
        return false;
    } else {
        states.erase(states.begin());
        return true;
    }
}

} // unonymous namespace

/****** PrefixTree *******/

void PrefixTree::create(const std::string& input_file, const std::string& output_file) {
    DecompressedPrefixTree decompressed_tree;
    std::ifstream reader(input_file);
    std::string token;
    while (getline(reader, token)) {
        decompressed_tree.add(token);
    }
    String tree_string = decompressed_tree.to_string();
    std::ofstream writer(output_file, std::ofstream::binary);
    writer.write((const char*)tree_string.c_str(), tree_string.size());
    writer.close();
}

PrefixTree::PrefixTree(const std::string& input_file) {
    std::ifstream reader(input_file, std::ifstream::binary);
    size_t size = get_file_size(input_file);
    content.resize(size);
    reader.read(const_cast<char*>((char*)content.c_str()), size);
}

size_t PrefixTree::match(const std::string& token) const {
    const unsigned char* pointer = content.data();
    const unsigned char* end_of_content = content.data() + content.size();
    size_t num_coincided = 0;
    while (num_coincided < token.length() && pointer < end_of_content) {
        const unsigned char content_letter = *pointer;
        const unsigned char token_letter = token[num_coincided];
        if (content_letter == 0) {
            return num_coincided;
        } else if (content_letter < 128) {
            if (token_letter == content_letter) {
                ++num_coincided;
                ++pointer;
            } else {
                return num_coincided;
            }
        } else if (token_letter + 128 == content_letter) {
            ++num_coincided;
            pointer += 256 * 256 * pointer[1] + 256 * pointer[2] + pointer[3];
        } else {
            pointer += 4;
        }
    }
    return num_coincided;
}

const unsigned char* PrefixTree::get_root() const {
    return content.data();
}

void PrefixTree::viterbi(const double* logits, size_t length, size_t num_hypos, std::vector<std::string>& output_tokens,
                         std::vector<double>& output_logits) const {
    std::set<ViterbiState> states;
    states.insert({ "", 0, 0, false });
    for (size_t step = 0; step < length; ++step) {
        std::set<ViterbiState> updated_states;
        for (const ViterbiState& state : states) {
            // fall from any state to terminate state
            double space_logit = state.logit + logits[NUM_LETTERS * step + to_int(' ')];
            if (pre_insert(step, updated_states, space_logit, num_hypos)) {
                updated_states.insert({ state.prefix, state.offset, space_logit, true });
            }
            unsigned char current_letter = content[state.offset];
            if (state.terminated || current_letter == 0) {
                continue;
            }
            // fall from non-terminate state to 1 child state
            if (current_letter < 128) {
                double updated_logit = state.logit + logits[NUM_LETTERS * step + to_int(current_letter)];
                if (pre_insert(step, updated_states, updated_logit, num_hypos)) {
                    updated_states.insert({ state.prefix + static_cast<char>(current_letter), state.offset + 1, updated_logit, false });
                }
            // fall from non-terminate state to many children state
            } else {
                ViterbiState transition_state = state;
                while (current_letter != 0) {
                    current_letter -= 128;
                    const unsigned char* pointer = content.data() + transition_state.offset;
                    double updated_logit = transition_state.logit + logits[NUM_LETTERS * step + to_int(current_letter)];
                    if (pre_insert(step, updated_states, updated_logit, num_hypos)) {
                        updated_states.insert({ transition_state.prefix + static_cast<char>(current_letter),
                                               transition_state.offset + 256 * 256 * pointer[1] + 256 * pointer[2] + pointer[3],
                                               updated_logit, false });
                    }
                    transition_state.offset += 4;
                    current_letter = content[transition_state.offset];
                }
            }
        }
        states = std::move(updated_states);
    }
    output_tokens.reserve(states.size());
    output_logits.reserve(states.size());
    for (auto iter = states.rbegin(); iter != states.rend(); ++iter) {
        output_tokens.emplace_back(std::move(iter->prefix));
        output_logits.push_back(iter->logit);
    }
}


/********* PrefixTreeAutomata ***********/

PrefixTreeAutomata::PrefixTreeAutomata(const PrefixTree& tree): state(tree.get_root()) {
}

size_t PrefixTreeAutomata::get_transitions(unsigned char* output) const {
    if (state == nullptr || *state == 0) {
        return 0;
    }
    if (*state < 128) {
        output[0] = *state;
        return 1;
    }
    size_t num_transitions = 0;
    const unsigned char* pointer = state;
    while (*pointer != 0) {
        output[num_transitions++] = *pointer - 128;
        pointer += 4;
    }
    return num_transitions;
}

void PrefixTreeAutomata::make_transition(unsigned char transition) {
    if (state == nullptr || *state == 0) {
        return;
    }
    if (*state < 128) {
        assert(*state == transition);
        ++state;
        return;
    }
    while (*state != 0) {
        if (*state == 128 + transition) {
            state += 256 * 256 * state[1] + 256 * state[2] + state[3];
            return;
        }
        state += 4;
    }
    assert(false);
}

/********* Python bidings *********/

extern "C" {

/*** PrefixTree ***/

std::unique_ptr<PrefixTree> prefix_tree;

void create_from_file(const char* file_name, size_t size) {
    prefix_tree = boost::make_unique<PrefixTree>(std::string(file_name, size));
}

void destroy() {
    prefix_tree.release();
}

size_t match(const char* token, size_t length) {
    return prefix_tree->match(std::string(token, length));
}

void viterbi(const double* logits, size_t length, size_t num_hypos, char* output, double* predictions) {
    std::vector<std::string> result_tokens;
    std::vector<double> result_logits;
    prefix_tree->viterbi(logits, length, num_hypos, result_tokens, result_logits);
    std::string concatenated_output;
    for (const std::string& token : result_tokens) {
        concatenated_output += token + "$";
    }
    if (concatenated_output.length() > 0) {
        std::memcpy(output, concatenated_output.c_str(), concatenated_output.length() - 1);
    }
    std::memcpy(predictions, result_logits.data(), sizeof(double) * result_logits.size());
}

/*** Prefix tree automata ***/

void* create_automata() {
    return new PrefixTreeAutomata(*prefix_tree);
}

void destroy_automata(void* automata) {
    delete (PrefixTreeAutomata*)automata;
}

size_t get_transitions(void* automata, unsigned char* output) {
    PrefixTreeAutomata& autom = *((PrefixTreeAutomata*)automata);
    return autom.get_transitions(output);
}

void make_transition(void* automata, unsigned char transition) {
    PrefixTreeAutomata& autom = *((PrefixTreeAutomata*)automata);
    return autom.make_transition(transition);
}


} // extern "C"
