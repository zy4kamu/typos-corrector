#include "prefix-tree.h"

#include <algorithm>

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* PrefixTreeNode */

PrefixTreeNode::PrefixTreeNode(const char* data): data(data) {
    if (data != nullptr) {
      uint8_t num_transitions = *reinterpret_cast<const uint8_t*>(data);
      transitions.resize(num_transitions);
      std::memcpy(transitions.data(), data + 1, num_transitions);
    }
}

PrefixTreeNode PrefixTreeNode::move(char letter) {
    auto found = std::find(transitions.begin(), transitions.end(), letter);
    if (found == transitions.end()) {
        return PrefixTreeNode();
    }
    size_t transition_index = std::distance(transitions.begin(), found);
    return PrefixTreeNode(data + *reinterpret_cast<const int32_t*>(data + 1 + transitions.size() + 4 * transition_index));
}

/* PrefixTree */

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

    root = PrefixTreeNode(static_cast<char*>(source));
}

PrefixTree::~PrefixTree() {
    if (munmap((void*)root.data, file_size) < 0) {
        perror("read_file: couldn't munmap");
    }
    close(file_desrciptor);
}

bool PrefixTree::check(const std::string& message) {
    return forward(message) != nullptr;
}

bool PrefixTree::get(const std::string& message, size_t limit, std::vector<std::string>& pretendents) {
    const char* pointer = forward(message);
    if (pointer == nullptr) {
        return false;
    }
    if (!walk(pointer, message, limit, pretendents)) {
        pretendents.clear();
    }
    return true;
}

const char* PrefixTree::forward(const std::string& message) const {
    PrefixTreeNode node = root;
    for (size_t i = 0; i < message.size(); ++i) {
        // run over transition letters and find current one
        size_t transition_index = std::string::npos;
        for (size_t j = 0; j < node.transitions.size(); ++j) {
            if (node.transitions[j] == message[i]) {
                transition_index = j;
                break;
            }
        }

        // return false if found nothing
        if (transition_index == std::string::npos) {
            return nullptr;
        }

        // move to next subree
        node = node.move(node.transitions[transition_index]);
    }
    return node.data;
}

bool PrefixTree::walk(const char* pointer, const std::string& prefix, size_t limit, std::vector<std::string>& pretendents) {
    uint8_t num_transitions = *reinterpret_cast<const uint8_t*>(pointer);
    if (pretendents.size() > limit) {
        return false;
    }
    if (num_transitions > 0) {
      for (uint8_t j = 0; j < num_transitions; ++j) {
          const char* next_pointer = pointer + *reinterpret_cast<const int32_t*>(pointer + 1 + num_transitions + 4 * j);
          const std::string next_prefix = prefix + pointer[1 + j];
          if (!walk(next_pointer, next_prefix, limit, pretendents)) {
              return false;
          }
      }
      return true;
    }
    pretendents.push_back(prefix);
    return pretendents.size() <= limit;
}
