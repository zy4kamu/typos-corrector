#pragma once

#include <vector>

#include <boost/filesystem/path.hpp>

#include "common.h"

namespace NNetworkHypoSearcher {

struct OpenCLConnector {
    OpenCLConnector();
    cl::Buffer read_buffer_from_file(const boost::filesystem::path& input_file, size_t size, int memory_permissions);
    void vector_matrix_multiply(const cl::Buffer& vector, const cl::Buffer& matrix, int_type matrix_num_rows,
                                int_type matrix_num_cols, cl::Buffer& output);
    void add_to_vector(const cl::Buffer& to_add, cl::Buffer& vector, int_type size);

    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Context               context;
    cl::CommandQueue          queue;
};

} // namespace NOpenCLConnector
