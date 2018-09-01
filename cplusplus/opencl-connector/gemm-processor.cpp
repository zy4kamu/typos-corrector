#include "gemm-processor.h"

#include <cassert>
#include <fstream>

#include "../utils/utils.h"

namespace NOpenCLConnector {

GEMMProcessor::GEMMProcessor(OpenCLConnector& opencl_connector, int_type num_rows, int_type num_cols):
    opencl_connector(opencl_connector), num_rows(num_rows), num_cols(num_cols) {
    assert(num_rows % 32 == 0);
    assert(num_cols % 32 == 0);
    intermediate_buffer = cl::Buffer(opencl_connector.context, CL_MEM_READ_WRITE,
                                     sizeof(float_type) * num_rows * num_cols / 32);

    // get source code
    std::ifstream reader(std::string(ROOT_DIRECTORY) + "/gemm-processor.cl");
    std::string src(std::istreambuf_iterator<char>(reader), (std::istreambuf_iterator<char>()));
    assert(src.size() > 0);
    sources = cl::Program::Sources(1, src);

    // build program
    program = cl::Program(opencl_connector.context, sources);
    int error = program.build();
    assert(error == 0);
    _unused(error);
}

void GEMMProcessor::vector_matrix_multiply(const cl::Buffer& vector, const cl::Buffer& matrix, cl::Buffer& output) {
    // kernel for intermediate_buffer calculation
    int error = 0;
    cl::Kernel intermediate_kernel = cl::Kernel(program, "intermediate_multipilcation", &error);
    assert(error == 0);
    _unused(error);
    intermediate_kernel.setArg(0, vector);
    intermediate_kernel.setArg(1, matrix);
    intermediate_kernel.setArg(2, num_rows);
    intermediate_kernel.setArg(3, num_cols);
    intermediate_kernel.setArg(4, intermediate_buffer);
    error = opencl_connector.queue.enqueueNDRangeKernel(intermediate_kernel, 0, num_rows * num_cols / 32, 1);
    assert(error == 0);

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

void GEMMProcessor::add_to_vector(const cl::Buffer& to_add, cl::Buffer& output, int_type size) {
    int error = 0;
    cl::Kernel add_to_vector_kernel = cl::Kernel(program, "add_to_vector", &error);
    assert(error == 0);
    _unused(error);
    add_to_vector_kernel.setArg(0, to_add);
    add_to_vector_kernel.setArg(1, output);
    error = opencl_connector.queue.enqueueNDRangeKernel(add_to_vector_kernel, 0, size, 1);
    assert(error == 0);
}

} // namespace NOpenCLConnector
