#pragma once

#include "opencl-connector.h"

namespace NOpenCLConnector {

class MatrixMultiplicator {
public:
    MatrixMultiplicator(OpenCLConnector& opencl_connector, int_type num_rows, int_type max_num_cols);
    void vector_matrix_multiply(const cl::Buffer& vector, const cl::Buffer& matrix, int_type num_cols, cl::Buffer& output);
private:
    OpenCLConnector&     opencl_connector;
    int_type             num_rows;
    int_type             max_num_cols;
    cl::Buffer           intermediate_buffer;
    cl::Program::Sources sources;
    cl::Program          program;
};

} // namespace NOpenCLConnector