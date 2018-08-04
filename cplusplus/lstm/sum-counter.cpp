#include "sum-counter.h"

#include <cassert>
#include <fstream>

SumCounter::SumCounter(size_t size, size_t local_size): size(size), local_size(local_size) {
    // Get platform
    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);

    // Get device
    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    assert(devices.size() > 0);
    device = devices.front();

    // Get source code
    std::ifstream reader(std::string(ROOT_DIRECTORY) + "/sum-counter.cl");
    std::string src(std::istreambuf_iterator<char>(reader), (std::istreambuf_iterator<char>()));
    assert(src.size() > 0);
    sources = cl::Program::Sources(1, src);

    // Build program
    context = cl::Context(device);
    program = cl::Program(context, sources);
    int error = program.build();
    assert(error == 0);

    // Create kernel
    device_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * size);
    kernel = cl::Kernel(program, "calculate_sum", &error);
    assert(error == 0);
    kernel.setArg(0, device_buffer);
    kernel.setArg(1, size);

    queue = cl::CommandQueue(context, device);

    std::vector<cl_int> data(size, 1);
    int error = queue.enqueueWriteBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_int) * size, data.data());
    assert(error == 0);
}

cl_int SumCounter::calculate(const std::vector<cl_int>& data) {
    assert(data.size() == size);
    int error = queue.enqueueWriteBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_int) * size, data.data());
    assert(error == 0);
    //queue.enqueueNDRangeKernel(kernel, 0, size, local_size, NULL);
    cl_int result = -1;
    error = queue.enqueueReadBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_int), &result);
    assert(error == 0);
    return result;
}
