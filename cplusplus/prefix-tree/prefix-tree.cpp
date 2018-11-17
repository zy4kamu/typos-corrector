#include "prefix-tree.h"

#include <algorithm>

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

PrefixTree::PrefixTree(const std::string& filename) {
    if ((file_desrciptor = open(filename.c_str(), O_RDONLY)) < 0) {
        perror("read_file: couldn't create file descriptor");
    }

    struct stat file_statistics;
    if (fstat(file_desrciptor, &file_statistics) < 0) {
        perror("read_file: couldn't find file size");
    }
    file_size = file_statistics.st_size;

    void* source;
    if ((source = mmap(0, file_size, PROT_READ, MAP_SHARED, file_desrciptor, 0)) == MAP_FAILED) {
        perror("read_file: couldn't use mmap to map file to pointer");
    }
    root = static_cast<const char*>(source);
}

PrefixTree::~PrefixTree() {
    if (munmap((void*)root, file_size) < 0) {
        perror("read_file: couldn't munmap");
    }
    close(file_desrciptor);
}

const std::vector<char>& PrefixTree::get_transitions() const {
    return state.transitions;
}

void PrefixTree::reset_pass() {
    reset_pass(root);
}

void PrefixTree::move(char letter) {
    auto found = std::find(state.transitions.begin(), state.transitions.end(), letter);
    if (found == state.transitions.end()) {
        reset_pass(nullptr);
    } else {
        size_t transition_index = std::distance(state.transitions.begin(), found);
        reset_pass(state.current_pointer + *reinterpret_cast<const int32_t*>(state.current_pointer + 1 + state.transitions.size() + 4 * transition_index));
    }
}

bool PrefixTree::check(const std::string& message) {
    for (size_t i = 0; i < message.size(); ++i) {
        // run over transition letters and find current one
        size_t transition_index = std::string::npos;
        for (size_t j = 0; j < state.transitions.size(); ++j) {
            if (state.transitions[j] == message[i]) {
                transition_index = j;
                break;
            }
        }

        // return false if found nothing
        if (transition_index == std::string::npos) {
            return false;
        }

        // move to next subree
        reset_pass(state.current_pointer + *reinterpret_cast<const int32_t*>(state.current_pointer + 1 + state.transitions.size() + 4 * transition_index));
    }
    return true;
}

const PrefixTreeState& PrefixTree::get_state() const {
    return state;
}

PrefixTreeState PrefixTree::move(const PrefixTreeState& state, char letter) {
    this->state = state;
    this->move(letter);
    return this->state;
}

void PrefixTree::reset_pass(const char* pointer) {
    // case of invalid pointer
    if (pointer == nullptr) {
        state.current_pointer = nullptr;
        state.transitions.clear();
        return;
    }

    // read transitions
    state.current_pointer = pointer;
    uint8_t num_transitions = *reinterpret_cast<const uint8_t*>(state.current_pointer);
    state.transitions.resize(num_transitions);
    std::memcpy(state.transitions.data(), state.current_pointer + 1, num_transitions);
}

