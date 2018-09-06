#include "common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>

#include "../utils/utils.h"

namespace NNetworkHypoSearcher {

const size_t MESSAGE_SIZE = 25;

std::vector<float_type> read_file(const std::string& filename) {
    std::vector<float_type> data;
    std::ifstream file(filename, std::ios::binary);
    float_type item;
    while (file.read(reinterpret_cast<char*>(&item), sizeof(float_type))) {
        data.push_back(item);
    }
    return data;
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
