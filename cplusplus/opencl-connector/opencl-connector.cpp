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

/*
"    for (int shift = 1; shift < 128; shift <<= 1) {"
"        if ((local_id + 1) % shift == 0) { buffer[local_id] += buffer[local_id / 2]; }              "
"        barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \n"
"    } "
*/

const char* PROGRAM_SOURCES =
"__kernel void intermediate_multipilcation(__global float* vector, __global float* matrix, int num_rows, int num_cols, \n"
"                                          __global float* output) {                                                   \n"
"    __local float buffer[128];        // MUST EQUAL TO local_size !!!                                                         \n"
"    __local float vector_buffer[128]; // MUST EQUAL TO local_size !!!                                                         \n"
"    __local float matrix_buffer[128]; // MUST EQUAL TO local_size !!!                                                         \n"

"    int local_id = get_local_id(0);                                                                                   \n"
"    int global_id = get_global_id(0);                                                                                 \n"
"    int local_size = get_local_size(0);                                                                               \n"
"    int row_index = global_id / num_cols;                                                                             \n"

"    vector_buffer[local_id] = vector[local_id];                                                                                   \n"
"    matrix_buffer[local_id] = matrix[global_id];                                                                                   \n"
"    buffer[local_id] = vector_buffer[local_id] * matrix_buffer[local_id];                                             \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                  // 0.003398 up to here                                                                                   \n"

"    for (int shift = local_size / 2; shift > 0; shift >>= 1) {                                                        \n"
"        if (local_id < shift) {                                                                                       \n"
"            buffer[local_id] += buffer[local_id + shift];                                                             \n"
"        }                                                                                                             \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \n"
"    }                                                            // 0.022732 up to here                                                                                                                  \n"

"    if (local_id == 0) {                                                                                              \n"
"        output[global_id / 128] = buffer[local_id];                                          \n"
"    }                                                            //  0.023751 up to here                                                                              \n"
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
    intermediate_kernel.setArg(4, output);
    cl::Event event;
    error = opencl_connector->queue.enqueueNDRangeKernel(intermediate_kernel, 0, num_rows * num_cols, 128, NULL, &event);
    assert(error == 0);
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
//    std::ifstream reader("/home/stepan/git-repos/typos-corrector/cplusplus/opencl-connector/matrix-multiplication.cl");
//    std::string src(std::istreambuf_iterator<char>(reader), (std::istreambuf_iterator<char>()));
//    assert(src.size() > 0);
//    src += "\n";
//    src += PROGRAM_SOURCES;
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
