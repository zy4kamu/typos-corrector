#include "prefix-tree.h"

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
    data = static_cast<char*>(source);
}

PrefixTree::~PrefixTree() {
    if (munmap((void*)data, file_size) < 0) {
        perror("read_file: couldn't munmap");
    }
    close(file_desrciptor);
}

bool PrefixTree::check(const std::string& message) {
    const char* pointer = data;
    for (size_t i = 0; i < message.size(); ++i) {
        // first byte contains number of transitions
        uint8_t num_transitions = *reinterpret_cast<const uint8_t*>(pointer);
        size_t transition_index = std::string::npos;

        // run over transition letters and find current one
        for (size_t j = 0; j < num_transitions; ++j) {
            if (pointer[1 + j] == message[i]) {
                transition_index = j;
                break;
            }
        }

        // return false if found nothing
        if (transition_index == std::string::npos) {
            return false;
        }

        // move to next subree
        pointer += *reinterpret_cast<const int32_t*>(pointer + 1 + num_transitions + 4 * transition_index);
    }
    return true;
}
