#include "opencl-connector.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include "common.h"

namespace NOpenCLConnector {

namespace {

std::vector<float_type> read_file(const std::string& filename) {
    std::vector<float_type> data;
    std::ifstream file(filename, std::ios::binary);
    float_type item;
    while (file.read(reinterpret_cast<char*>(&item), sizeof(float_type))) {
        data.push_back(item);
    }
    return data;
}

const char* PROGRAM_SOURCES =
"__kernel void intermediate_multipilcation(__global float* vector, __global float* matrix, int num_rows, int num_cols, \n"
"                                          __global float* output) {                                                   \n"
"    int global_id = get_global_id(0);                                                                                 \n"
"    int row_index = global_id / num_cols;                                                                             \n"
"    int col_index = global_id % num_cols;                                                                             \n"
"    vector += 32 * row_index;                                                                                         \n"
"    matrix += 32 * row_index * num_cols + col_index;                                                                  \n"
"    float value = 0;                                                                                                  \n"
"    for (int i = 0; i < 32; ++i) {                                                                                    \n"
"        value += vector[i] * matrix[i * num_cols];                                                                    \n"
"    }                                                                                                                 \n"
"    output[row_index * num_cols + col_index] = value;                                                                 \n"
"}                                                                                                                     \n"
"                                                                                                                      \n"
"__kernel void final_sum(__global float* buffer, int num_rows, int num_cols, __global float* output) {                 \n"
"    int col_index = get_global_id(0);                                                                                 \n"
"    float value = 0;                                                                                                  \n"
"    for (int i = 0; i < num_rows; ++i) {                                                                              \n"
"        value += buffer[i * num_cols + col_index];                                                                    \n"
"    }                                                                                                                 \n"
"    output[col_index] = value;                                                                                        \n"
"}                                                                                                                     \n"
"                                                                                                                      \n"
"__kernel void add_to_vector(__global float* to_add, __global float* output) {                                         \n"
"    int global_id = get_global_id(0);                                                                                 \n"
"    output[global_id] += to_add[global_id];                                                                           \n"
"}                                                                                                                     \n";

#define _unused(x) ((void)(x))

} // anonymous namespace

// Matrix Mutliplicator class

MatrixMultiplicator::MatrixMultiplicator(OpenCLConnector& opencl_connector, const cl::Program& program, int_type num_rows,
                                         int_type num_cols):
    opencl_connector(&opencl_connector), program(&program), num_rows(num_rows), num_cols(num_cols) {
    // intermediate_buffer
    assert(num_rows % 32 == 0);
    assert(num_cols % 32 == 0);
    intermediate_buffer = cl::Buffer(opencl_connector.context, CL_MEM_READ_WRITE,
                                     sizeof(float_type) * num_rows * num_cols / 32);
}

void MatrixMultiplicator::operator()(const cl::Buffer& vector, const cl::Buffer& matrix, cl::Buffer& output) {
    clock_t begin = clock();

    // intermediate_kernel
    int error = 0;
    cl::Kernel intermediate_kernel = cl::Kernel(*program, "intermediate_multipilcation", &error);
    assert(error == 0);
    _unused(error);
    intermediate_kernel.setArg(0, vector);
    intermediate_kernel.setArg(1, matrix);
    intermediate_kernel.setArg(2, num_rows);
    intermediate_kernel.setArg(3, num_cols);
    intermediate_kernel.setArg(4, intermediate_buffer);
    error = opencl_connector->queue.enqueueNDRangeKernel(intermediate_kernel, 0, num_rows * num_cols / 32, 1);
    assert(error == 0);

    // final_sum_kernel
    cl::Kernel final_sum_kernel = cl::Kernel(*program, "final_sum", &error);
    assert(error == 0);
    final_sum_kernel.setArg(0, intermediate_buffer);
    final_sum_kernel.setArg(1, num_rows / 32);
    final_sum_kernel.setArg(2, num_cols);
    final_sum_kernel.setArg(3, output);
    cl::Event event;
    error = opencl_connector->queue.enqueueNDRangeKernel(final_sum_kernel, 0, num_cols, 1, NULL, &event);
    event.wait();
    assert(error == 0);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "multiplied matrix in " << elapsed_secs << " (" << num_rows << ", " << num_cols << ")\n";
}

// OpenCLConnector class

OpenCLConnector::OpenCLConnector() {
    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);

    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    assert(devices.size() > 0);
    device = devices.front();

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    // build program from sources
    sources = cl::Program::Sources(1, PROGRAM_SOURCES);
    program = cl::Program(context, sources);
    int error = program.build();
    assert(error == 0);
    _unused(error);
}

cl::Buffer OpenCLConnector::read_buffer_from_file(const std::string& input_file, size_t size,
                                                  int memory_permissions) {
    // TODO: memory map directly to GPU
    cl::Buffer buffer(context, memory_permissions, sizeof(float_type) * size);
    std::vector<float_type> data = read_file(input_file);
    assert(data.size() == size);
    int error = queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(float_type) * size, data.data());
    assert(error == 0);
    _unused(error);
    return buffer;
}

MatrixMultiplicator OpenCLConnector::create_matrix_multiplicator(int_type num_rows, int_type num_cols) {
    return MatrixMultiplicator(*this, program, num_rows, num_cols);
}

void OpenCLConnector::add_to_vector(const cl::Buffer& to_add, cl::Buffer& output, int_type size) {
    int error = 0;
    cl::Kernel add_to_vector_kernel = cl::Kernel(program, "add_to_vector", &error);
    assert(error == 0);
    _unused(error);
    add_to_vector_kernel.setArg(0, to_add);
    add_to_vector_kernel.setArg(1, output);
    error = queue.enqueueNDRangeKernel(add_to_vector_kernel, 0, size, 1);
    assert(error == 0);
}

} // namespace NOpenCLConnector
