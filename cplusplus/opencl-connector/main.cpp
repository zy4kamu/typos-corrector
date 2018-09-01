#include "opencl-connector.h"
#include "gemm-processor.h"

#include <iostream>
#include <vector>

using float_type = NOpenCLConnector::float_type;
using int_type = NOpenCLConnector::int_type;

int main()
{
    int_type num_rows = 512;
    int_type num_cols = 32;
    NOpenCLConnector::OpenCLConnector connector;

    std::vector<float_type> vector(num_rows, 1);
    cl::Buffer vector_buffer(connector.context, CL_MEM_READ_WRITE, sizeof(float_type) * num_rows);
    connector.queue.enqueueWriteBuffer(vector_buffer, CL_TRUE, 0, sizeof(float_type) * num_rows, vector.data());

    std::vector<float_type> matrix(num_rows * num_cols);
    for (int_type i = 0; i < num_rows; ++i) {
        for (int_type j = 0; j < num_cols; ++j) {
            matrix[num_cols * i + j] = i;
        }
    }
    cl::Buffer matrix_buffer(connector.context, CL_MEM_READ_WRITE, sizeof(float_type) * num_rows * num_cols);
    connector.queue.enqueueWriteBuffer(matrix_buffer, CL_TRUE, 0, sizeof(float_type) * num_rows * num_cols, matrix.data());

    cl::Buffer output_buffer(connector.context, CL_MEM_READ_WRITE, sizeof(float_type) * num_cols);
    NOpenCLConnector::MatrixMultiplicator multiplicator = connector.create_matrix_multiplicator(num_rows, num_cols);
    multiplicator(vector_buffer, matrix_buffer, output_buffer);
    std::vector<float_type> output(num_cols);
    connector.queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float_type) * num_cols, output.data());
    std::cout << "matrix multiply: ";
    for (const float_type item : output) {
        std::cout << item << " ";
    }
    std::cout << std::endl;

    connector.add_to_vector(output_buffer, output_buffer, num_cols);
    connector.queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float_type) * num_cols, output.data());
    std::cout << "vector sum: ";
    for (const float_type item : output) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
