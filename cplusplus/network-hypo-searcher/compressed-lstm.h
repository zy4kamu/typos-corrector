#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../opencl-connector/opencl-connector.h"
#include "common.h"

namespace NNetworkHypoSearcher {

using OpenCLConnector = NOpenCLConnector::OpenCLConnector;

class CompressedLSTMCell {
public:
    CompressedLSTMCell(OpenCLConnector& connector, const std::string& input_folder, const std::vector<std::string>& models);
    void process(size_t one_hot_index, size_t model_index = 0);
    void get_output(std::vector<float_type>& output) const;
    int_type get_input_size() const { return input_size; }
    int_type get_output_size() const { return lstm_size; }
    const cl::Buffer& get_hidden_buffer() const { return hidden_buffer; }
    void make_all_buffers_zero();
    void store_current_hypo_pass();
    void reset_current_hypo_pass();
private:
    void calculate_ijfo(int_type one_hot_index, size_t model_index);

    // sizes of the model
    int_type                input_size;
    int_type                compressor_size;
    int_type                lstm_size;

    // Usual OpenCL routine
    OpenCLConnector         opencl_connector;
    cl::Program::Sources    sources;
    cl::Program             program;

    // Matrix multiplications buffers: input_and_hidden_buffer -> ijfo_buffer
    cl::Buffer              input_and_hidden_buffer; // glued together to calculate ijfo
    cl::Buffer              input_buffer;            // doesn't own memory
    cl_buffer_region        input_buffer_region;     // used in creation of input_buffer only
    cl::Buffer              hidden_buffer;           // doesn't own memory
    cl_buffer_region        hidden_buffer_region;    // used in creation of hidden_buffer only
    std::vector<cl::Buffer> left_matrix_buffers;
    std::vector<cl::Buffer> intermediate_matrix_buffers;
    std::vector<cl::Buffer> right_matrix_buffers;
    std::vector<cl::Buffer> bias_buffers;

    // buffers for lstm_cell
    cl::Buffer              state_buffer;
    cl::Buffer              ijfo_buffer;

    // kernels
    cl::Kernel              lstm_cell_kernel;
    cl::Kernel              initialize_buffers_kernel;
    cl::Kernel              set_input_kernel;
    cl::Kernel              reset_current_hypo_pass_kernel;
    cl::Kernel              store_current_hypo_pass_kernel;

    // kernels for reset between trying different hypos
    cl::Buffer              stored_state_buffer;
    cl::Buffer              stored_hidden_buffer;
};

} // namespace NOpenCLConnector
