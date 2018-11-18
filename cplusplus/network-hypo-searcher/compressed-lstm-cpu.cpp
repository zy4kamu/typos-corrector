#include "compressed-lstm-cpu.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <tuple>

#include "utils.h"

namespace NNetworkHypoSearcher {

namespace {

float_type sigmoid(float_type value) {
    return 1. / (1. + exponent(-value));
}

float_type hyperbolic_tangent(float_type value) {
    float_type exp_activation = exponent(-2.0 * value);
    return (1. - exp_activation) / (1. + exp_activation);
}

} // anonymous namespace

CompressedLSTMCellCPU::CompressedLSTMCellCPU(const std::string& input_folder, const std::vector<std::string>& models):
    input_folder(input_folder), models(models), is_model_loaded(false) {
}

void CompressedLSTMCellCPU::load() {
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
            input_and_hidden_buffer.resize(input_size + lstm_size, 0);
            input_buffer = input_and_hidden_buffer.data();
            hidden_buffer = input_and_hidden_buffer.data() + input_size;
            state_buffer.resize(lstm_size, 0);
            ijfo_buffer.resize(4 * lstm_size, 0);
            stored_state_buffer.resize(lstm_size);
            stored_hidden_buffer.resize(lstm_size);
        } else {
            assert(lstm_size == local_lstm_size);
            assert(compressor_size == local_compressor_size);
            assert(input_size == local_input_size);
        }

        left_matrix_buffers.push_back(read_file(model_prefix + "left_matrix"));
        intermediate_matrix_buffers.emplace_back(compressor_size);
        right_matrix_buffers.push_back(read_file(model_prefix + "right_matrix"));
        bias_buffers.push_back(read_file(model_prefix + "bias"));
    }
    is_model_loaded = true;
}

void CompressedLSTMCellCPU::unload() {
    if (is_model_loaded) {
        input_and_hidden_buffer.clear();
        input_buffer = nullptr;
        hidden_buffer = nullptr;
        left_matrix_buffers.clear();
        intermediate_matrix_buffers.clear();
        right_matrix_buffers.clear();
        bias_buffers.clear();
        state_buffer.clear();
        ijfo_buffer.clear();
        // intentionally do not clear stored_state_buffer and stored_hidden_buffer: it will accelerate next pass twice
    }
    is_model_loaded = false;
}

bool CompressedLSTMCellCPU::is_loaded() const {
    return is_model_loaded;
}

void CompressedLSTMCellCPU::process(size_t one_hot_index, size_t model_index) {
    calculate_ijfo(one_hot_index, model_index);

    for (int_type i = 0; i < lstm_size; ++i) {
        // unpack gates
        float_type input_gate      = ijfo_buffer[i];
        float_type activation_gate = ijfo_buffer[lstm_size + i];
        float_type forget_gate     = ijfo_buffer[2 * lstm_size + i];
        float_type output_gate     = ijfo_buffer[3 * lstm_size + i];

        // forget information
        state_buffer[i] *= sigmoid(1. +  forget_gate);
        state_buffer[i] += sigmoid(input_gate) * hyperbolic_tangent(activation_gate);
        hidden_buffer[i] = hyperbolic_tangent(state_buffer[i]) * sigmoid(output_gate);
    }
}

void CompressedLSTMCellCPU::calculate_ijfo(int_type one_hot_index, size_t model_index) {
    assert(model_index < left_matrix_buffers.size());
    std::vector<float_type>& left_matrix_buffer         = left_matrix_buffers[model_index];
    std::vector<float_type>& intermediate_matrix_buffer = intermediate_matrix_buffers[model_index];
    std::vector<float_type>& right_matrix_buffer        = right_matrix_buffers[model_index];
    std::vector<float_type>& bias_buffer                = bias_buffers[model_index];

    std::memset(input_buffer, 0, input_size * sizeof(float_type));
    input_buffer[one_hot_index] = 1;

    vector_matrix_multiply(input_and_hidden_buffer.data(), left_matrix_buffer.data(), input_size + lstm_size,
                           compressor_size, intermediate_matrix_buffer.data());
    vector_matrix_multiply(intermediate_matrix_buffer.data(), right_matrix_buffer.data(), compressor_size,
                           4 * lstm_size, ijfo_buffer.data());
    add_to_vector(bias_buffer.data(), ijfo_buffer.data(), 4 * lstm_size);
}

void CompressedLSTMCellCPU::make_all_buffers_zero() {
    std::fill_n(hidden_buffer, lstm_size, 0);
    std::fill_n(state_buffer.begin(), lstm_size, 0);
}

void CompressedLSTMCellCPU::store_current_hypo_pass() {
    memcpy(stored_state_buffer.data(), state_buffer.data(), lstm_size * sizeof(float_type));
    memcpy(stored_hidden_buffer.data(), hidden_buffer, lstm_size * sizeof(float_type));
}

void CompressedLSTMCellCPU::restore_current_hypo_pass() {
    memcpy(state_buffer.data(), stored_state_buffer.data(), lstm_size * sizeof(float_type));
    memcpy(hidden_buffer, stored_hidden_buffer.data(), lstm_size * sizeof(float_type));
}

CompressedLSTMCellCPU::InternalState CompressedLSTMCellCPU::get_internal_state() const {
    std::vector<float_type> hidden_buffer_copy(lstm_size);
    memcpy(hidden_buffer_copy.data(), hidden_buffer, lstm_size * sizeof(float_type));
    return std::make_pair(state_buffer, std::move(hidden_buffer_copy));
}

void CompressedLSTMCellCPU::set_internal_state(const InternalState& state) {
    memcpy(state_buffer.data(), state.first.data(), lstm_size * sizeof(float_type));
    memcpy(hidden_buffer, state.second.data(), lstm_size * sizeof(float_type));
}

} // namespace NNetworkHypoSearcher
