#include "compressed_lstm.h"
#include <cblas.h>

#include <iostream>
#include <fstream>

void save_binary_vector(const std::string& file, const std::vector<cl_float>& data) {
    std::ofstream writer(file, std::ios::binary);
    writer.write((char*)data.data(), data.size() * sizeof(cl_float));
}

void dummy_test() {
    // size_t input_size = 3;
    // size_t compressor_size = 2;
    // size_t lstm_size = 1;

    std::string output_folder = "/tmp/parameters/";
    system("mkdir -p /tmp/parameters");
    std::vector<cl_float> left_matrix = { .1, .2,
                                          .3, .4,
                                          .5, .6,
                                          .7, .8 };
    save_binary_vector(output_folder + "left_matrix", left_matrix);
    std::vector<cl_float> right_matrix = { .1, .2, .3, .4,
                                           .5, .6, .7, .8 };
    save_binary_vector(output_folder + "right_matrix", right_matrix);
    std::vector<cl_float> bias = { 1, 2, 3, 4 };
    save_binary_vector(output_folder + "bias", bias);

    CompressedLSTMCell cell(output_folder, 3, 2, 1);
    std::vector<cl_float> input = { .1, .2, .3 };
    std::vector<cl_float> output = { 0 };
    cell.process(input, output);
    std::cout << output[0] << std::endl;
}

void real_test() {
    CompressedLSTMCell cell("/home/stepan/git-repos/typos-corrector/python/model/parameters/encode_lstm_", 27, 128, 512);
    std::vector<cl_float> input(27, 0);
    input[0] = 1;
    std::vector<cl_float> output(512);
    cell.process(input, output);
    for (cl_float _ : output) std::cout << _ << " ";
    std::cout << std::endl;
}

int main() {
    real_test();
}
