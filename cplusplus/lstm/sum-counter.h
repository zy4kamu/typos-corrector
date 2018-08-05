#pragma once

#include <cl2.hpp>
#include <memory>
#include <vector>

class SumCounter {
public:
    SumCounter(cl_int size, cl_int local_size = 1);
    cl_float calculate(const std::vector<cl_float>& data);
    std::vector<cl_float>& exp(std::vector<cl_float>& data);
private:
    cl_float                  size;
    cl_float                    local_size;

    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Program::Sources      sources;
    cl::Context               context;
    cl::Program               program;
    cl::Buffer                device_buffer;
    cl::Kernel                sum_kernel;
    cl::Kernel                exp_kernel;
    cl::CommandQueue          queue;
};
