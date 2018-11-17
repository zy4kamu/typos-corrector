#pragma once

#include "compressed-lstm-cpu.h"

#include <string>
#include <vector>

namespace NNetworkHypoSearcher {

class NetworkAutomataCPU {
public:
    NetworkAutomataCPU(const std::string& input_folder);
    void encode_message(const std::string& messsage, std::vector<float_type>& first_letter_logits);
    void load();
    void unload();
    bool is_loaded() const;
    void reset_pass();
    void apply(char letter, std::vector<float_type>& next_letter_logits);
    CompressedLSTMCellCPU::InternalState get_internal_state() const;
    void set_internal_state(const CompressedLSTMCellCPU::InternalState& state);
private:
    void get_output(std::vector<float_type>& output);

    std::string             input_folder;
    CompressedLSTMCellCPU   lstm;
    std::vector<float_type> hidden_layer_weights;
    std::vector<float_type> hidden_layer_bias;
};

} // namespace NNetworkHypoSearcher
