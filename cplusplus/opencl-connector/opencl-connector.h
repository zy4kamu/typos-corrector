#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace NOpenCLConnector {

struct OpenCLConnector {
    OpenCLConnector();
    cl::Buffer read_buffer_from_file(const std::string& input_file, size_t size, int memory_permissions);
    void add_to_vector(const cl::Buffer& to_add, cl::Buffer& vector, int_type size);

    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Context               context;
    cl::CommandQueue          queue;
};

} // namespace NOpenCLConnector
