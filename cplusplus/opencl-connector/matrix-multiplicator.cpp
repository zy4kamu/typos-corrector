#include "matrix-multiplicator.h"

#include <cassert>
#include <fstream>

#include "../utils/utils.h"

namespace NOpenCLConnector {

MatrixMultiplicator::MatrixMultiplicator(OpenCLConnector& opencl_connector, int_type num_rows, int_type max_num_cols):
    opencl_connector(opencl_connector), num_rows(num_rows), max_num_cols(max_num_cols) {
    assert(num_rows % 32 == 0);
    assert(max_num_cols % 32 == 0);
    intermediate_buffer = cl::Buffer(opencl_connector.context, CL_MEM_READ_WRITE,
                                     sizeof(float_type) * num_rows * max_num_cols / 32);

    // get source code
    std::ifstream reader(std::string(ROOT_DIRECTORY) + "/matrix-multiplicator.cl");
    std::string src(std::istreambuf_iterator<char>(reader), (std::istreambuf_iterator<char>()));
    assert(src.size() > 0);
    sources = cl::Program::Sources(1, src);

    // build program
    program = cl::Program(opencl_connector.context, sources);
    int error = program.build();
    assert(error == 0);
    _unused(error);
}

void MatrixMultiplicator::vector_matrix_multiply(const cl::Buffer& vector, const cl::Buffer& matrix, int_type num_cols,
                                                 cl::Buffer& output) {
    // kernel for intermediate_buffer calculation
    int error = 0;
    cl::Kernel intermediate_kernel = cl::Kernel(program, "intermediate_multipilcation", &error);
    assert(error == 0);
    intermediate_kernel.setArg(0, vector);
    intermediate_kernel.setArg(1, matrix);
    intermediate_kernel.setArg(2, num_rows);
    intermediate_kernel.setArg(3, num_cols);
    intermediate_kernel.setArg(4, intermediate_buffer);
    error = opencl_connector.queue.enqueueNDRangeKernel(intermediate_kernel, 0, num_rows * num_cols / 32, 1);
    assert(error == 0);
    _unused(error);

    // kernel for final sum calculation
    cl::Kernel final_sum_kernel = cl::Kernel(program, "final_sum", &error);
    assert(error == 0);
    final_sum_kernel.setArg(0, intermediate_buffer);
    final_sum_kernel.setArg(1, num_rows / 32);
    final_sum_kernel.setArg(2, num_cols);
    final_sum_kernel.setArg(3, output);
    error = opencl_connector.queue.enqueueNDRangeKernel(final_sum_kernel, 0, num_cols, 1);
    assert(error == 0);
}

} // namespace NOpenCLConnector
