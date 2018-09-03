#include "compressed-lstm-gpu.h"

#include <cassert>
#include <iostream>

#include <boost/make_unique.hpp>

#include "common.h"
#include "../utils/utils.h"

namespace NNetworkHypoSearcher {

namespace {

const char* PROGRAM_SOURCES =
"__kernel void reset(__global float* hidden_buffer, __global float* state_buffer,                                   \n"
"                    int input_size, int lstm_size) {                                                               \n"
"    int global_id = get_global_id(0);                                                                              \n"
"    hidden_buffer[global_id] = 0;                                                                                  \n"
"    state_buffer[global_id] = 0;                                                                                   \n"
"}                                                                                                                  \n"
"                                                                                                                   \n"
"__kernel void copy_hidden_and_state_buffers(__global float* to_hidden_buffer, __global float* from_hidden_buffer,  \n"
"                                            __global float* to_state_buffer, __global float* from_state_buffer) {  \n"
"    int global_id = get_global_id(0);                                                                              \n"
"    to_hidden_buffer[global_id] = from_hidden_buffer[global_id];                                                   \n"
"    to_state_buffer[global_id] = from_state_buffer[global_id];                                                     \n"
"}                                                                                                                  \n"
"                                                                                                                   \n"
"__kernel void set_input(__global float* input_buffer, int one_hot_index) {                                         \n"
"    int global_id = get_global_id(0);                                                                              \n"
"    if (global_id == one_hot_index) {                                                                              \n"
"        input_buffer[global_id] = 1;                                                                               \n"
"    } else {                                                                                                       \n"
"        input_buffer[global_id] = 0;                                                                               \n"
"    }                                                                                                              \n"
"}                                                                                                                  \n"
"                                                                                                                   \n"
"float sigmoid(float value) {                                                                                       \n"
"    return 1. / (1. + exp(-value));                                                                                \n"
"}                                                                                                                  \n"
"                                                                                                                   \n"
"float hyperbolic_tangent(float value) {                                                                            \n"
"    float exp_activation = exp(-2.0 * value);                                                                      \n"
"    return (1. - exp_activation) / (1. + exp_activation);                                                          \n"
"}                                                                                                                  \n"
"                                                                                                                   \n"
"__kernel void lstm_cell(__global float* hidden_buffer, __global float* state_buffer,                               \n"
"                        __global float* ijfo_buffer, int input_size, int lstm_size)                                \n"
"{                                                                                                                  \n"
"    if (get_global_id(0) == 0) { printf(\"ZZZZZZZZZZ %lf\\n\", ijfo_buffer[0]); }                                       \n"
"    // unpack gates                                                                                                \n"
"    size_t global_id = get_global_id(0);                                                                           \n"
"    float input_gate      = *(ijfo_buffer                 + global_id);                                            \n"
"    float activation_gate = *(ijfo_buffer +     lstm_size + global_id);                                            \n"
"    float forget_gate     = *(ijfo_buffer + 2 * lstm_size + global_id);                                            \n"
"    float output_gate     = *(ijfo_buffer + 3 * lstm_size + global_id);                                            \n"
"                                                                                                                   \n"
"    // forget information                                                                                          \n"
"    __global float* state = state_buffer + global_id;                                                              \n"
"    *state *= sigmoid(1. + forget_gate);                                                                           \n"
"                                                                                                                   \n"
"    // update information                                                                                          \n"
"    *state += sigmoid(input_gate) * hyperbolic_tangent(activation_gate);                                           \n"
"                                                                                                                   \n"
"    // update output                                                                                               \n"
"    __global float* hidden = hidden_buffer + global_id;                                                            \n"
"    *hidden = hyperbolic_tangent(*state) * sigmoid(output_gate);                                                   \n"
"}                                                                                                                  \n";

} // anonymous namespace

CompressedLSTMCellGPU::CompressedLSTMCellGPU(OpenCLConnector& opencl_connector,
                                       const std::string& input_folder,
                                       const std::vector<std::string>& models)
    : opencl_connector(opencl_connector) {

    // read parameters from file
    for (size_t i = 0; i < models.size(); ++i) {
        //  get sizes
        const std::string model_prefix = input_folder + "/" + models[i];
        int_type local_lstm_size = static_cast<int_type>(get_file_size(model_prefix + "bias")) / (4 * sizeof(float_type));
        int_type local_compressor_size = static_cast<int_type>(get_file_size(model_prefix + "right_matrix") /
                                                           (4 * sizeof(float_type) * local_lstm_size));
        int_type local_input_size = static_cast<int_type>(get_file_size(model_prefix + "left_matrix") /
                                                      (sizeof(float_type) * local_compressor_size)) - local_lstm_size;
        if (i == 0) {
            lstm_size = local_lstm_size;
            compressor_size = local_compressor_size;
            input_size = local_input_size;

            // create buffers
            // TODO: experiment with memory access types
            input_and_hidden_buffer = cl::Buffer(opencl_connector.context, CL_MEM_READ_WRITE,
                                                 sizeof(float_type) * (input_size + lstm_size));
            state_buffer = cl::Buffer(opencl_connector.context, CL_MEM_WRITE_ONLY, sizeof(float_type) * lstm_size);
            ijfo_buffer = cl::Buffer(opencl_connector.context, CL_MEM_WRITE_ONLY, 4 * sizeof(float_type) * lstm_size);

            // hidden_buffer is a sub buffer of input_and_hidden_buffer and is created for simplicity
            hidden_buffer_region.origin = sizeof(float_type) * input_size;
            hidden_buffer_region.size = sizeof(float_type) * lstm_size;
            hidden_buffer = input_and_hidden_buffer.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                                                    &hidden_buffer_region);

            // hidden_buffer is a sub buffer of input_and_hidden_buffer and is created for simplicity
            input_buffer_region.origin = 0;
            input_buffer_region.size = sizeof(float_type) * input_size;
            input_buffer = input_and_hidden_buffer.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                                                   &input_buffer_region);

            // buffers for reset between trying different hypos
            stored_state_buffer = cl::Buffer(opencl_connector.context, CL_MEM_WRITE_ONLY, sizeof(float_type) * lstm_size);
            stored_hidden_buffer = cl::Buffer(opencl_connector.context, CL_MEM_WRITE_ONLY, sizeof(float_type) * lstm_size);
        } else {
            assert(lstm_size == local_lstm_size);
            assert(compressor_size == local_compressor_size);
            assert(input_size == local_input_size);
        }

        left_matrix_buffers.emplace_back(opencl_connector.read_buffer_from_file(model_prefix + "left_matrix",
                                                                                (input_size + lstm_size) * compressor_size,
                                                                                CL_MEM_WRITE_ONLY));
        intermediate_matrix_buffers.emplace_back(opencl_connector.context, CL_MEM_WRITE_ONLY,
                                                 sizeof(float_type) * compressor_size);
        right_matrix_buffers.emplace_back(opencl_connector.read_buffer_from_file(model_prefix + "right_matrix",
                                                                                compressor_size * 4 * lstm_size,
                                                                                CL_MEM_WRITE_ONLY));
        bias_buffers.emplace_back(opencl_connector.read_buffer_from_file(model_prefix + "bias",
                                                                         4 * lstm_size,
                                                                         CL_MEM_WRITE_ONLY));
    }

    // build program from sources
    sources = cl::Program::Sources(1, PROGRAM_SOURCES);
    program = cl::Program(opencl_connector.context, sources);
    int error = program.build();
    assert(error == 0);
    _unused(error);

    // create kernel for lstm cell computation
    lstm_cell_kernel = cl::Kernel(program, "lstm_cell", &error);
    assert(error == 0);
    lstm_cell_kernel.setArg(0, hidden_buffer);
    lstm_cell_kernel.setArg(1, state_buffer);
    lstm_cell_kernel.setArg(2, ijfo_buffer);
    lstm_cell_kernel.setArg(3, input_size);
    lstm_cell_kernel.setArg(4, lstm_size);

    // create kernel for reseting lstm after each start
    error = 0;
    initialize_buffers_kernel = cl::Kernel(program, "reset", &error);
    assert(error == 0);
    initialize_buffers_kernel.setArg(0, hidden_buffer);
    initialize_buffers_kernel.setArg(1, state_buffer);
    initialize_buffers_kernel.setArg(2, input_size);
    initialize_buffers_kernel.setArg(3, lstm_size);

    // create kernel for setup input on each lstm_cell call
    error = 0;
    set_input_kernel = cl::Kernel(program, "set_input", &error);
    assert(error == 0);
    set_input_kernel.setArg(0, input_buffer);

    // create kernel for reseting current hypo pass
    error = 0;
    reset_current_hypo_pass_kernel = cl::Kernel(program, "copy_hidden_and_state_buffers", &error);
    assert(error == 0);
    reset_current_hypo_pass_kernel.setArg(0, hidden_buffer);
    reset_current_hypo_pass_kernel.setArg(1, stored_hidden_buffer);
    reset_current_hypo_pass_kernel.setArg(2, state_buffer);
    reset_current_hypo_pass_kernel.setArg(3, stored_state_buffer);

    // create kernel for saving current hypo pass
    error = 0;
    store_current_hypo_pass_kernel = cl::Kernel(program, "copy_hidden_and_state_buffers", &error);
    assert(error == 0);
    store_current_hypo_pass_kernel.setArg(0, stored_hidden_buffer);
    store_current_hypo_pass_kernel.setArg(1, hidden_buffer);
    store_current_hypo_pass_kernel.setArg(2, stored_state_buffer);
    store_current_hypo_pass_kernel.setArg(3, state_buffer);
    store_current_hypo_pass_kernel.setArg(4, lstm_size);

    // create matrix multiplicators
    first_matrix_multiplicator = opencl_connector.create_matrix_multiplicator(input_size + lstm_size, compressor_size);
    second_matrix_multiplicator = opencl_connector.create_matrix_multiplicator(compressor_size, 4 * lstm_size);
}

void CompressedLSTMCellGPU::make_all_buffers_zero() {
    int error = opencl_connector.queue.enqueueNDRangeKernel(initialize_buffers_kernel, 0, lstm_size, 1);
    assert(error == 0);
    _unused(error);
}

void CompressedLSTMCellGPU::store_current_hypo_pass() {
    int error = opencl_connector.queue.enqueueNDRangeKernel(store_current_hypo_pass_kernel, 0, lstm_size, 1);
    assert(error == 0);
    _unused(error);
}

void CompressedLSTMCellGPU::reset_current_hypo_pass() {
    int error = opencl_connector.queue.enqueueNDRangeKernel(reset_current_hypo_pass_kernel, 0, lstm_size, 1);
    assert(error == 0);
    _unused(error);
}

void CompressedLSTMCellGPU::get_output(std::vector<float_type>& output) const {
    assert(static_cast<float_type>(output.size()) == lstm_size);
    int error = opencl_connector.queue.enqueueReadBuffer(hidden_buffer, CL_TRUE, 0, sizeof(float_type) * lstm_size,
                                                         output.data());
    std::cout << output[0] << std::endl;
    assert(error == 0);
    _unused(error);
}

void CompressedLSTMCellGPU::process(size_t one_hot_index, size_t model_index) {
    std::cout << "AAAAAAAAAAA " << one_hot_index << std::endl;
    calculate_ijfo(one_hot_index, model_index);
    int error = opencl_connector.queue.enqueueNDRangeKernel(lstm_cell_kernel, 0, lstm_size, 1);
    assert(error == 0);
    _unused(error);
}

void CompressedLSTMCellGPU::calculate_ijfo(int_type one_hot_index, size_t model_index) {
    assert(model_index < left_matrix_buffers.size());
    cl::Buffer& left_matrix_buffer = left_matrix_buffers[model_index];
    cl::Buffer& intermediate_matrix_buffer = intermediate_matrix_buffers[model_index];
    cl::Buffer& right_matrix_buffer = right_matrix_buffers[model_index];
    cl::Buffer& bias_buffer = bias_buffers[model_index];

    // set input
    set_input_kernel.setArg(1, one_hot_index);
    int error = opencl_connector.queue.enqueueNDRangeKernel(set_input_kernel, 0, input_size, 1);
    assert(error == 0);
    _unused(error);

    // calculate ijfo
    first_matrix_multiplicator(input_and_hidden_buffer, left_matrix_buffer, intermediate_matrix_buffer);
    second_matrix_multiplicator(intermediate_matrix_buffer, right_matrix_buffer, ijfo_buffer);
    opencl_connector.add_to_vector(bias_buffer, ijfo_buffer, 4 * lstm_size);
}

} // namespace NNetworkHypoSearcher
