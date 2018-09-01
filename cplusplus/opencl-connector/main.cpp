#include "opencl-connector.h"
#include "matrix-multiplicator.h"

#include <iostream>
#include <vector>

using float_type = NOpenCLConnector::float_type;

int main()
{
    NOpenCLConnector::OpenCLConnector connector;

    std::vector<float_type> vector(32, 1);
    cl::Buffer vector_buffer(connector.context, CL_MEM_READ_WRITE, sizeof(float_type) * 32);
    connector.queue.enqueueWriteBuffer(vector_buffer, CL_TRUE, 0, sizeof(float_type) * 32, vector.data());

    std::vector<float_type> matrix(32 * 32);
    for (size_t i = 0; i < 32; ++i) {
        for (size_t j = 0; j < 32; ++j) {
            matrix[32 * i + j] = i;
        }
    }
    cl::Buffer matrix_buffer(connector.context, CL_MEM_READ_WRITE, sizeof(float_type) * 32 * 32);
    connector.queue.enqueueWriteBuffer(matrix_buffer, CL_TRUE, 0, sizeof(float_type) * 32 * 32, matrix.data());

    cl::Buffer output_buffer(connector.context, CL_MEM_READ_WRITE, sizeof(float_type) * 32);
    NOpenCLConnector::MatrixMultiplicator multiplicator(connector, 32, 32);
    multiplicator.vector_matrix_multiply(vector_buffer, matrix_buffer, 32, output_buffer);
    std::vector<float_type> output(32);
    connector.queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float_type) * 32, output.data());
    for (const float_type item : output) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
