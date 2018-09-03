#pragma once

#include "compressed-lstm-cpu.h"

#include <string>
#include <vector>

namespace NNetworkHypoSearcher {

class NetworkAutomataCPU {
public:
    NetworkAutomataCPU(const std::string& input_folder);
    void encode_message(const std::string& messsage, std::vector<float_type>& first_letter_logits);
    void reset();
    void apply(char letter, std::vector<float_type>& next_letter_logits);
private:
    void get_output(std::vector<float_type>& output);

    CompressedLSTMCellCPU   lstm;
    std::vector<float_type> hidden_layer_weights;
    std::vector<float_type> hidden_layer_bias;
};

} // namespace NNetworkHypoSearcher
