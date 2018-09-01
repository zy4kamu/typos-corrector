#pragma once

#include "compressed-lstm.h"
#include "../opencl-connector/gemm-processor.h"
#include "../opencl-connector/opencl-connector.h"

#include <string>
#include <vector>

#include "common.h"

namespace NNetworkHypoSearcher {

class NetworkAutomata {
public:
    NetworkAutomata(const std::string& input_folder);
    void encode_message(const std::string& messsage, std::vector<float_type>& first_letter_logits);
    void reset();
    void apply(char letter, std::vector<float_type>& next_letter_logits);
private:
    void get_output(std::vector<float_type>& first_letter_logits);
    OpenCLConnector       opencl_connector;
    cl::Program::Sources  sources;
    cl::Program           program;
    cl::Kernel            logits_to_probabilities_kernel;

    CompressedLSTMCell lstm;
    cl::Buffer         hidden_layer_weights;
    cl::Buffer         hidden_layer_bias;
    cl::Buffer         output;

    GEMMProcessor gemm_processor;
};

} // namespace NOpenCLConnector
