#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common.h"

namespace NNetworkHypoSearcher {


class CompressedLSTMCellCPU {
public:
    using InternalState = std::pair<std::vector<float_type>, std::vector<float_type>>;
    CompressedLSTMCellCPU(const std::string& input_folder, const std::vector<std::string>& models);
    void load();
    void unload();
    bool is_loaded() const;
    void process(size_t one_hot_index, size_t model_index = 0);
    const float_type* get_output() const { return hidden_buffer; }
    int_type get_input_size() const { return input_size; }
    int_type get_output_size() const { return lstm_size; }
    void make_all_buffers_zero();
    void store_current_hypo_pass();
    void restore_current_hypo_pass();
    InternalState get_internal_state() const;
    void set_internal_state(const InternalState& state);
private:
    void calculate_ijfo(int_type one_hot_index, size_t model_index);

    // parameters required to load model from the file
    std::string              input_folder;
    std::vector<std::string> models;
    bool is_model_loaded;

    // sizes of the model
    int_type                input_size;
    int_type                compressor_size;
    int_type                lstm_size;

    // Matrix multiplications buffers: input_and_hidden_buffer -> ijfo_buffer
    std::vector<float_type>              input_and_hidden_buffer; // glued together to calculate ijfo
    float_type*                          input_buffer;            // doesn't own memory
    float_type*                          hidden_buffer;           // doesn't own memory
    std::vector<std::vector<float_type>> left_matrix_buffers;
    std::vector<std::vector<float_type>> intermediate_matrix_buffers;
    std::vector<std::vector<float_type>> right_matrix_buffers;
    std::vector<std::vector<float_type>> bias_buffers;

    // buffers for lstm_cell
    std::vector<float_type> state_buffer;
    std::vector<float_type> ijfo_buffer;

    // kernels for reset between trying different hypos
    std::vector<float_type> stored_state_buffer;
    std::vector<float_type> stored_hidden_buffer;
};

} // namespace NNetworkHypoSearcher
