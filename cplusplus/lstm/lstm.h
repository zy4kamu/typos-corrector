#pragma once

#include <cl2.hpp>
#include <memory>
#include <string>
#include <vector>

class LSTMCell {
public:
    LSTMCell(const std::string& input_folder, cl_int input_size, cl_int compressor_size, cl_int lstm_size);
    void process(const std::vector<cl_float>& input, std::vector<cl_float>& output);
    std::vector<cl_float>& exp(std::vector<cl_float>& data);
private:
    cl_float                  input_size;
    cl_float                  compressor_size;
    cl_float                  lstm_size;

    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Program::Sources      sources;
    cl::Context               context;
    cl::Program               program;
    cl::CommandQueue          queue;

    cl::Kernel                lstm_cell_kernel;
    cl::Buffer                input_and_hidden_buffer; // glued together to calculate ijfo
    cl::Buffer                state_buffer;
    cl::Buffer                left_matrix_kernel_buffer;
    cl::Buffer                right_matrix_kernel_buffer;
    cl::Buffer                bias_kernel_buffer;
    cl::Buffer                ijfo_buffer;

    cl::Kernel                exp_kernel;
    cl::Buffer                device_buffer;
};
