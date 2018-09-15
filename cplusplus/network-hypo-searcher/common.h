#pragma once

#include <cstring>
#include <string>
#include <vector>

#ifdef USE_OPENCL
    #include "../opencl-connector/common.h"
#endif

namespace NNetworkHypoSearcher {

#ifdef USE_OPENCL
using float_type = NOpenCLConnector::float_type;
using int_type = cl_int;
#else
    using float_type = float;
    using int_type = int;
#endif
extern const size_t MESSAGE_SIZE;

std::vector<float_type> read_file(const std::string& filename);
void vector_matrix_multiply(const float_type* vector, const float_type* matrix, size_t num_rows, size_t num_cols,
                            float_type* output);
void add_to_vector(const float_type* to_add, float_type* output, size_t size);
int get_letter(const std::string& message, size_t position);
float_type exponent(float_type value);

} // namespace NNetworkHypoSearcher
