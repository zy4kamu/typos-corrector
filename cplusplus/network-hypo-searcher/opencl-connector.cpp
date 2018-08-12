#include "opencl-connector.h"

#include <cassert>
#include <clBLAS.h>
#include <fstream>

#include "common.h"

namespace {

std::vector<float_type> read_file(const boost::filesystem::path& filename) {
    std::vector<float_type> data;
    std::ifstream file(filename.string(), std::ios::binary);
    float_type item;
    while (file.read(reinterpret_cast<char*>(&item), sizeof(float_type))) {
        data.push_back(item);
    }
    return data;
}

} // anonymous namespace

OpenCLConnector::OpenCLConnector() {
    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);

    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    assert(devices.size() > 0);
    device = devices.front();

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    // setup clBLAS
    int error = clblasSetup();
    assert(error == CL_SUCCESS);
}

cl::Buffer OpenCLConnector::read_buffer_from_file(const boost::filesystem::path& input_file, size_t size,
                                                  int memory_permissions) {
    // TODO: memory map directly to GPU
    cl::Buffer buffer(context, memory_permissions, sizeof(float_type) * size);
    std::vector<float_type> data = read_file(input_file);
    assert(data.size() == size);
    int error = queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(float_type) * size, data.data());
    assert(error == 0);
    return buffer;
}

void OpenCLConnector::vector_matrix_multiply(const cl::Buffer& vector, const cl::Buffer& matrix, int_type matrix_num_rows,
                                             int_type matrix_num_cols, cl::Buffer& output) {
    cl_command_queue local_queue = queue.get();
    clblasStatus status =
    clblasSgemv(clblasRowMajor,  // order
                clblasTrans,     // transA
                matrix_num_rows, // M
                matrix_num_cols, // N
                1.f,             // alpha
                matrix.get(),    // A
                0,               // offA
                matrix_num_cols, // lda
                vector.get(),    // x
                0,               // offx
                1,               // incx
                0,               // beta
                output.get(),    // y
                0,               // offy
                1,               // incy
                1,               // numCommandQueues
                &local_queue,    // commandQueues
                0,               // numEventsInWaitList
                NULL,            // eventWaitList
                NULL);           // events
    assert(status == clblasSuccess);
}

void OpenCLConnector::add_to_vector(const cl::Buffer& to_add, cl::Buffer& vector, int_type size) {
    cl_command_queue local_queue = queue.get();
    int status =
    clblasSaxpy(size,         // N
                1,            // alpha
                to_add.get(), // X
                0,            // offx
                1,            // incx
                vector.get(), // Y
                0,            // offy
                1,            // incy
                1,            // numCommandQueues
                &local_queue, // commandQueues
                0,            // numEventsInWaitList
                NULL,         // eventWaitList
                NULL);        // events
    assert(status == clblasSuccess);
}
