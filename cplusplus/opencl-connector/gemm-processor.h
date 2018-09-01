#pragma once

#include "opencl-connector.h"

namespace NOpenCLConnector {

class GEMMProcessor {
public:
    GEMMProcessor(OpenCLConnector& opencl_connector, int_type num_rows, int_type num_cols);
    void vector_matrix_multiply(const cl::Buffer& vector, const cl::Buffer& matrix, cl::Buffer& output);
    void add_to_vector(const cl::Buffer& to_add, cl::Buffer& output, int_type size);
private:
    OpenCLConnector&     opencl_connector;
    int_type             num_rows;
    int_type             num_cols;
    cl::Buffer           intermediate_buffer;
    cl::Program::Sources sources;
    cl::Program          program;
};

} // namespace NOpenCLConnector
