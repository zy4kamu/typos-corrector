#include "network-automata-gpu.h"
#include "../utils/utils.h"

#include <cassert>

#include "common.h"

namespace NNetworkHypoSearcher {

namespace {

const char* PROGRAM_SOURCES =
"__kernel void initialize_logits(__global float* logits) {           \n"
"    size_t global_id = get_global_id(0);                            \n"
"    logits[global_id] = -1024;                                      \n"
"}                                                                   \n"
"                                                                    \n"
"// NOTE: this kernel must be called with 1 local group of size 32   \n"
"__kernel void logits_to_probabilities(__global float* buffer) {     \n"
"    size_t global_id = get_global_id(0);                            \n"
"                                                                    \n"
"    // create local buffer                                          \n"
"    __local float sum_buffer[32];                                   \n"
"                                                                    \n"
"    // exponent all logits                                          \n"
"    buffer[global_id] = exp(buffer[global_id]);                     \n"
"                                                                    \n"
"    // get normalization factor                                     \n"
"    sum_buffer[global_id] = buffer[global_id];                      \n"
"    for (int shift = 16; shift > 0; shift >>= 1) {                  \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                               \n"
"        if (global_id < shift) {                                    \n"
"            sum_buffer[global_id] += sum_buffer[global_id + shift]; \n"
"        }                                                           \n"
"    }                                                               \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                   \n"
"    float normalization_factor = sum_buffer[0];                     \n"
"                                                                    \n"
"    // get probabilities                                            \n"
"    buffer[global_id] /= normalization_factor;                      \n"
"}                                                                   \n";

} // anonymous namespace

NetworkAutomataGPU::NetworkAutomataGPU(const std::string& input_folder)
    : lstm(opencl_connector, input_folder, { "encode_lstm_", "decode_lstm_" })
    , hidden_layer_weights(opencl_connector.read_buffer_from_file(input_folder + "/hidden_layer_weights",
                                                                  lstm.get_output_size() * NUM_LETTERS,
                                                                  CL_MEM_WRITE_ONLY))
    , hidden_layer_bias(opencl_connector.read_buffer_from_file(input_folder + "/hidden_layer_bias",
                                                               NUM_LETTERS,
                                                               CL_MEM_WRITE_ONLY))
    , output(opencl_connector.context, CL_MEM_WRITE_ONLY, sizeof(float_type) * NUM_LETTERS) {

    // build program from sources
    sources = cl::Program::Sources(1, PROGRAM_SOURCES);
    program = cl::Program(opencl_connector.context, sources);
    int error = program.build();
    assert(error == 0);

    // initialize logits
    cl::Kernel initialize_logits_kernel = cl::Kernel(program, "initialize_logits", &error);
    assert(error == 0);
    initialize_logits_kernel.setArg(0, output);
    error = opencl_connector.queue.enqueueNDRangeKernel(initialize_logits_kernel, 0, NUM_LETTERS, 1);
    assert(error == 0);

    // create kernel for lstm cell computation
    logits_to_probabilities_kernel = cl::Kernel(program, "logits_to_probabilities", &error);
    assert(error == 0);
    logits_to_probabilities_kernel.setArg(0, output);

    matrix_multiplicator = opencl_connector.create_matrix_multiplicator(lstm.get_output_size(), NUM_LETTERS);
}

void NetworkAutomataGPU::encode_message(const std::string& message, std::vector<float_type>& first_letter_logits) {
    lstm.make_all_buffers_zero();
    for (size_t i = 0; i < MESSAGE_SIZE; ++i) {
        lstm.process(get_letter(message, MESSAGE_SIZE - i - 1), 0);
    }
    lstm.store_current_hypo_pass();
    get_output(first_letter_logits);
}

void NetworkAutomataGPU::reset() {
    lstm.reset_current_hypo_pass();
}

void NetworkAutomataGPU::apply(char letter, std::vector<float_type>& next_letter_logits) {
    lstm.process(to_int(letter), 1);
    get_output(next_letter_logits);
}

void NetworkAutomataGPU::get_output(std::vector<float_type>& output_logits) {
    // linear transform of lstm output
    matrix_multiplicator(lstm.get_hidden_buffer(), hidden_layer_weights, output);
    opencl_connector.add_to_vector(hidden_layer_bias, output, NUM_LETTERS);

    // logits to probabilities
    int error = opencl_connector.queue.enqueueNDRangeKernel(logits_to_probabilities_kernel, 0, NUM_LETTERS,
                                                            NUM_LETTERS);
    assert(error == 0);
    _unused(error);

    // read output
    error = opencl_connector.queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(float_type) * NUM_LETTERS,
                                                     output_logits.data());
    assert(error == 0);
}

} // namespace NNetworkHypoSearcher
