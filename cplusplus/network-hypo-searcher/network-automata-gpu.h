#pragma once

#include "compressed-lstm-gpu.h"
#include "../opencl-connector/opencl-connector.h"

#include <string>
#include <vector>

#include "common.h"

namespace NNetworkHypoSearcher {

class NetworkAutomataGPU {
public:
    NetworkAutomataGPU(const std::string& input_folder);
    void encode_message(const std::string& messsage, std::vector<float_type>& first_letter_logits);
    void reset();
    void apply(char letter, std::vector<float_type>& next_letter_logits);
private:
    void get_output(std::vector<float_type>& output_logits);
    OpenCLConnector       opencl_connector;
    cl::Program::Sources  sources;
    cl::Program           program;
    cl::Kernel            logits_to_probabilities_kernel;

    CompressedLSTMCellGPU lstm;
    cl::Buffer            hidden_layer_weights;
    cl::Buffer            hidden_layer_bias;
    cl::Buffer            output;

    NOpenCLConnector::MatrixMultiplicator matrix_multiplicator;
};

} // namespace NNetworkHypoSearcher
