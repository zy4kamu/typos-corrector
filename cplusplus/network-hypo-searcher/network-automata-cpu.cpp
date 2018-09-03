#include "network-automata-cpu.h"
#include "../utils/utils.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "common.h"

namespace NNetworkHypoSearcher {

namespace {


} // anonymous namespace

NetworkAutomataCPU::NetworkAutomataCPU(const std::string& input_folder)
    : lstm(input_folder, { "encode_lstm_", "decode_lstm_" })
    , hidden_layer_weights(read_file(input_folder + "/hidden_layer_weights"))
    , hidden_layer_bias(read_file(input_folder + "/hidden_layer_bias")) {
}

void NetworkAutomataCPU::encode_message(const std::string& message, std::vector<float_type>& first_letter_logits) {
    lstm.make_all_buffers_zero();
    for (size_t i = 0; i < MESSAGE_SIZE; ++i) {
        lstm.process(get_letter(message, MESSAGE_SIZE - i - 1), 0);
    }
    lstm.store_current_hypo_pass();
    get_output(first_letter_logits);
}

void NetworkAutomataCPU::reset() {
    lstm.reset_current_hypo_pass();
}

void NetworkAutomataCPU::apply(char letter, std::vector<float_type>& next_letter_logits) {
    lstm.process(to_int(letter), 1);
    get_output(next_letter_logits);
}

void NetworkAutomataCPU::get_output(std::vector<float_type>& output) {
    // linear transform of lstm output
    vector_matrix_multiply(lstm.get_output(), hidden_layer_weights.data(), lstm.get_output_size(), NUM_LETTERS,
                           output.data());
    add_to_vector(hidden_layer_bias.data(), output.data(), NUM_LETTERS);

    // logits to probabilities
    float_type max_item = std::accumulate(output.begin(), output.end(), std::numeric_limits<float_type>::min(),
                                          [](float_type first, float_type second) { return std::max(first, second); });
    std::transform(output.begin(), output.end(), output.begin(),
                   [max_item](float_type item) { return exponent(item - max_item); });
    float_type normalize_factor = std::accumulate(output.begin(), output.end(), 0,
                   [](float_type first, float_type second ) { return first + second; });
    std::transform(output.begin(), output.end(), output.begin(),
                   [normalize_factor](float_type item) { return item / normalize_factor; });

    // for (const float_type item : output) { std::cout << item << " "; }; std::cout << std::endl;
}

} // namespace NNetworkHypoSearcher
