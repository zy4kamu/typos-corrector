#include "common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../utils/utils.h"

namespace NNetworkHypoSearcher {

const size_t MESSAGE_SIZE = 25;

std::vector<float_type> read_file(const std::string& filename) {
    int file_desrciptor;
    if ((file_desrciptor = open(filename.c_str(), O_RDONLY)) < 0) {
        perror("read_file: couldn't create file descriptor");
    }

    struct stat file_statistics;
    if (fstat(file_desrciptor, &file_statistics) < 0) {
        perror("read_file: couldn't find file size");
    }
    size_t file_size = file_statistics.st_size;

    void *source;
    if ((source = mmap(0, file_size, PROT_READ, MAP_SHARED, file_desrciptor, 0)) == MAP_FAILED) {
        perror("read_file: couldn't use mmap to map file to pointer");
    }

    std::vector<float_type> content(file_size / sizeof(float_type));
    memcpy(content.data(), source, file_size);

    if (munmap(source, file_size) < 0) {
        perror("read_file: couldn't munmap");
    }
    close(file_desrciptor);
    return content;
}

void vector_matrix_multiply(const float_type* vector, const float_type* matrix, size_t num_rows, size_t num_cols,
                            float_type* output) {
    memset(output, 0, sizeof(float_type) * num_cols);
    for (size_t i = 0; i < num_rows; ++i) {
        float_type vector_value = vector[i];
        const float_type* matrix_row = matrix + i * num_cols;
        for (size_t j = 0; j < num_cols; ++j) {
            output[j] += vector_value * matrix_row[j];
        }
    }
}

void add_to_vector(const float_type* to_add, float_type* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] += to_add[i];
    }
}

int get_letter(const std::string& message, size_t position) {
    return position < message.length() ? to_int(message[position]) : to_int(' ');
}

float_type exponent(float_type value) {
    value = value > 8 ? 8 : value;
    return std::exp(value);
}

} // namespace NNetworkHypoSearcher
