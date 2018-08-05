#include "sum-counter.h"

#include <cassert>
#include <fstream>

SumCounter::SumCounter(cl_int size, cl_int local_size): size(size), local_size(local_size) {
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

    // Create sum_kernel
    device_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size);
    sum_kernel = cl::Kernel(program, "calculate_sum", &error);
    assert(error == 0);
    sum_kernel.setArg(0, device_buffer);
    sum_kernel.setArg(1, size);

    // Create exp_kernel
    exp_kernel = cl::Kernel(program, "calculate_exp", &error);
    assert(error == 0);
    exp_kernel.setArg(0, device_buffer);
    exp_kernel.setArg(1, size);

    queue = cl::CommandQueue(context, device);
}

cl_float SumCounter::calculate(const std::vector<cl_float>& data) {
    assert(static_cast<cl_float>(data.size()) == size);
    int error = queue.enqueueWriteBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_float) * size, data.data());
    assert(error == 0);
    error = queue.enqueueNDRangeKernel(sum_kernel, 0, size, local_size, NULL);
    assert(error == 0);
    cl_float result = -1;
    error = queue.enqueueReadBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_float), &result);
    assert(error == 0);
    return result;
}

std::vector<cl_float>& SumCounter::exp(std::vector<cl_float>& data) {
    assert(static_cast<cl_float>(data.size()) == size);
    int error = queue.enqueueWriteBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_float) * size, data.data());
    assert(error == 0);
    error = queue.enqueueNDRangeKernel(exp_kernel, 0, size, local_size, NULL);
    assert(error == 0);
    error = queue.enqueueReadBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_float) * size, data.data());
    assert(error == 0);
    return data;
}
