#pragma once

#include <cl2.hpp>
#include <memory>
#include <string>
#include <vector>

class CompressedLSTMCell {
public:
    CompressedLSTMCell(const std::string& input_folder_prefix, cl_int input_size, cl_int compressor_size, cl_int lstm_size);
    void process(const std::vector<cl_float>& input, std::vector<cl_float>& output);
private:
    void reset();
    void calculate_ijfo(const std::vector<cl_float>& input);

    // sizes of the model
    cl_int                    input_size;
    cl_int                    compressor_size;
    cl_int                    lstm_size;

    // Usual OpenCL routine
    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Program::Sources      sources;
    cl::Context               context;
    cl::Program               program;
    cl::CommandQueue          queue;

    // Matrix multiplications buffers: input_and_hidden_buffer -> ijfo_buffer
    cl::Buffer                input_and_hidden_buffer; // glued together to calculate ijfo
    cl::Buffer                left_matrix_kernel_buffer;
    cl::Buffer                intermediate_matrix_buffer;
    cl::Buffer                right_matrix_kernel_buffer;
    cl::Buffer                bias_kernel_buffer;

    // buffers for lstm_cell
    cl::Kernel                lstm_cell_kernel;
    cl::Buffer                state_buffer;
    cl::Buffer                ijfo_buffer;
};
