#include "prefix-tree.h"

#include <algorithm>

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/**** PrefixTreeMaster ****/

PrefixTreeMaster::PrefixTreeMaster(const std::string& filename) {
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
    initial_state.reset_pass(root);
}

PrefixTreeMaster::~PrefixTreeMaster() {
    if (munmap((void*)root, file_size) < 0) {
        perror("read_file: couldn't munmap");
    }
    close(file_desrciptor);
}

const PrefixTreeState& PrefixTreeMaster::get_initial_state() {
    return initial_state;
}

/**** PrefixTreeState ****/

const std::vector<char>& PrefixTreeState::get_transitions() const {
    return transitions;
}

void PrefixTreeState::move(char letter) {
    auto found = std::find(transitions.begin(), transitions.end(), letter);
    if (found == transitions.end()) {
        reset_pass(nullptr);
    } else {
        size_t transition_index = std::distance(transitions.begin(), found);
        reset_pass(current_pointer + *reinterpret_cast<const int32_t*>(current_pointer + 1 + transitions.size() + 4 * transition_index));
    }
}

PrefixTreeState PrefixTreeState::move(char letter) const {
    PrefixTreeState state(*this);
    state.move(letter);
    return state;
}

void PrefixTreeState::reset_pass(const char* pointer) {
    // case of invalid pointer
    if (pointer == nullptr) {
        current_pointer = nullptr;
        transitions.clear();
        return;
    }

    // read transitions
    current_pointer = pointer;
    uint8_t num_transitions = *reinterpret_cast<const uint8_t*>(current_pointer);
    transitions.resize(num_transitions);
    std::memcpy(transitions.data(), current_pointer + 1, num_transitions);
}

