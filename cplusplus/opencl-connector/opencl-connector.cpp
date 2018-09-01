#include "opencl-connector.h"

#include <cassert>
#include <fstream>

#include "common.h"
#include "clBLAS.h"

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

#define _unused(x) ((void)(x))

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

} // namespace NOpenCLConnector
