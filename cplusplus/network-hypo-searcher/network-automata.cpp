#include "network-automata.h"
#include "../utils/utils.h"

// TODO: namespaces to each project
// TODO: don't like naming conventions in this file (LOCAL_GROUP_SIZE is bad name, for example)

#include <cassert>
#include <fstream>

namespace {
const size_t MESSAGE_SIZE = 25;
const size_t LOCAL_GROUP_SIZE = 32;

int get_letter(const std::string& message, size_t position) {
    return position < message.length() ? to_int(message[position]) : to_int(' ');
}

} // anonymous namespace

NetworkAutomata::NetworkAutomata(const boost::filesystem::path& input_folder)
    : lstm(opencl_connector, input_folder, { "encode_lstm_", "decode_lstm_" })
    , hidden_layer_weights(opencl_connector.read_buffer_from_file(input_folder / "hidden_layer_weights",
                                                                  lstm.get_output_size() * NUM_LETTERS,
                                                                  CL_MEM_WRITE_ONLY))
    , hidden_layer_bias(opencl_connector.read_buffer_from_file(input_folder / "hidden_layer_bias",
                                                               NUM_LETTERS,
                                                               CL_MEM_WRITE_ONLY))
    // TODO: creation of buffer can be passed to OpenCLConnector
    , output(opencl_connector.context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * LOCAL_GROUP_SIZE) {

    // get source code
    std::ifstream reader(std::string(ROOT_DIRECTORY) + "/network-automata.cl");
    std::string src(std::istreambuf_iterator<char>(reader), (std::istreambuf_iterator<char>()));
    assert(src.size() > 0);
    sources = cl::Program::Sources(1, src);

    // build program
    program = cl::Program(opencl_connector.context, sources);
    int error = program.build();
    assert(error == 0);

    // initialize logits
    cl::Kernel initialize_logits_kernel = cl::Kernel(program, "initialize_logits", &error);
    assert(error == 0);
    initialize_logits_kernel.setArg(0, output);
    error = opencl_connector.queue.enqueueNDRangeKernel(initialize_logits_kernel, 0, LOCAL_GROUP_SIZE, 1);
    assert(error == 0);

    // create kernel for lstm cell computation
    logits_to_probabilities_kernel = cl::Kernel(program, "logits_to_probabilities", &error);
    assert(error == 0);
    logits_to_probabilities_kernel.setArg(0, output);
}

void NetworkAutomata::encode_message(const std::string& message, std::vector<cl_float>& first_letter_logits) {
    for (size_t i = 0; i < MESSAGE_SIZE; ++i) {
        lstm.process(get_letter(message, MESSAGE_SIZE - i - 1), 0);
    }
    get_output(first_letter_logits);
}

void NetworkAutomata::apply(char letter, std::vector<cl_float>& next_letter_logits) {
    lstm.process(to_int(letter), 1);
    get_output(next_letter_logits);
}

void NetworkAutomata::get_output(std::vector<cl_float>& first_letter_logits) {
    // linear transform of lstm output
    opencl_connector.vector_matrix_multiply(lstm.get_hidden_buffer(), hidden_layer_weights,
                                            lstm.get_output_size(), NUM_LETTERS, output);
    opencl_connector.add_to_vector(hidden_layer_bias, output, NUM_LETTERS);

    // logits to probabilities
    int error = opencl_connector.queue.enqueueNDRangeKernel(logits_to_probabilities_kernel, 0, LOCAL_GROUP_SIZE,
                                                            LOCAL_GROUP_SIZE);
    assert(error == 0);

    // read output
    // TODO: enqueueReadBuffer can be wrapped into OpenCLConnector
    error = opencl_connector.queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(cl_float) * NUM_LETTERS,
                                                     first_letter_logits.data());
    assert(error == 0);
}
