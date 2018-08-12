#pragma once

#include "compressed-lstm.h"
#include "opencl-connector.h"

#include <boost/filesystem/path.hpp>
#include <cl2.hpp>
#include <vector>

class NetworkAutomata {
public:
    NetworkAutomata(const boost::filesystem::path& input_folder);
    void encode_message(const std::string& messsage, std::vector<cl_float>& first_letter_logits);
    void apply(char letter, std::vector<cl_float>& next_letter_logits);
private:
    void get_output(std::vector<cl_float>& first_letter_logits);
    OpenCLConnector       opencl_connector;
    cl::Program::Sources  sources;
    cl::Program           program;
    cl::Kernel            logits_to_probabilities_kernel;

    CompressedLSTMCell lstm;
    cl::Buffer         hidden_layer_weights;
    cl::Buffer         hidden_layer_bias;
    cl::Buffer         output;
};
