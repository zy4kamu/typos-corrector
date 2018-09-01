#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace NOpenCLConnector {

class OpenCLConnector;

class MatrixMultiplicator {
public:
    MatrixMultiplicator() = default;
    void operator()(const cl::Buffer& vector, const cl::Buffer& matrix, cl::Buffer& output);
private:
    MatrixMultiplicator(OpenCLConnector& opencl_connector, const cl::Program& program, int_type num_rows, int_type num_cols);

    OpenCLConnector*   opencl_connector;
    const cl::Program* program;
    int_type           num_rows;
    int_type           num_cols;
    cl::Buffer         intermediate_buffer;

    friend class OpenCLConnector;
};

struct OpenCLConnector {
    OpenCLConnector();
    cl::Buffer read_buffer_from_file(const std::string& input_file, size_t size, int memory_permissions);
    MatrixMultiplicator create_matrix_multiplicator(int_type num_rows, int_type num_cols);
    void add_to_vector(const cl::Buffer& to_add, cl::Buffer& output, int_type size);

    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Context               context;
    cl::CommandQueue          queue;
private:
    cl::Program::Sources      sources;
    cl::Program               program;
};

} // namespace NOpenCLConnector
