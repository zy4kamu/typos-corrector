#pragma once

#include <cl2.hpp>
#include <memory>
#include <vector>

class SumCounter {
public:
    SumCounter(cl_int size, cl_int local_size = 1);
    cl_int calculate(const std::vector<cl_int>& data);
private:
    cl_int                    size;
    cl_int                    local_size;

    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Program::Sources      sources;
    cl::Context               context;
    cl::Program               program;
    cl::Buffer                device_buffer;
    cl::Kernel                kernel;
    cl::CommandQueue          queue;
};
