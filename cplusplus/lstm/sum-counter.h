#pragma once

#include <cl2.hpp>
#include <memory>
#include <vector>

class SumCounter {
public:
    SumCounter(size_t size, size_t local_size = 256);
    cl_int calculate(const std::vector<cl_int>& data);
private:
    size_t               size;
    size_t               local_size;

    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Program::Sources      sources;
    cl::Context               context;
    cl::Program               program;
    cl::Buffer                device_buffer;
    cl::Kernel                kernel;
    cl::CommandQueue          queue;
};
