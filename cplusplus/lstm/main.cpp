#include "lstm.h"
#include <cblas.h>

#include <iostream>
#include <fstream>

void save_binary_vector(const std::string& file, const std::vector<cl_float>& data) {
    std::ofstream writer(file, std::ios::binary);
    writer.write((char*)data.data(), data.size() * sizeof(cl_float));
}

void create_dummy_lstm(const std::string& output_folder) {
    // size_t input_size = 3;
    // size_t compressor_size = 2;
    // size_t lstm_size = 1;
    std::vector<cl_float> left_matrix = { 1, 2,
                                          3, 4,
                                          5, 6,
                                          7, 8 };
    save_binary_vector(output_folder + "left_matrix", left_matrix);
    std::vector<cl_float> right_matrix = { 1, 2, 3, 4,
                                           5, 6, 7, 8 };
    save_binary_vector(output_folder + "right_matrix", right_matrix);
    std::vector<cl_float> bias = { 1, 2, 3, 4 };
    save_binary_vector(output_folder + "bias", bias);
}

int main() {
    system("mkdir -p /tmp/parameters");
    create_dummy_lstm("/tmp/parameters/");
    LSTMCell cell("/tmp/parameters", 3, 2, 1);
    std::vector<cl_float> input = { 1, 2, 3 };
    std::vector<cl_float> output = { 0 };
    cell.process(input, output);
    std::cout << output[0] << std::endl;
}
