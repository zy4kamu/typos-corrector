#include "compressed-lstm.h"
#include "network-automata.h"
#include "opencl-connector.h"
#include "../utils/utils.h"

#include <cblas.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>

#include "common.h"

size_t argmax(const std::vector<float_type>& data) {
    float_type max = std::numeric_limits<float_type>::min();
    size_t best_index = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] > max) {
            max = data[i];
            best_index = i;
        }
    }
    return best_index;
}

void test_network_automata() {
    NetworkAutomata automata("/home/stepan/git-repos/typos-corrector/python/model/parameters/");
    std::vector<float_type> probabilities(27);
    automata.encode_message("masterdamweg", probabilities);
    for (size_t i = 0; i < 25; ++i) {
        size_t index = argmax(probabilities);
        char ch = to_char(static_cast<int32_t>(index));
        std::cout << ch;
        automata.apply(ch, probabilities);
    }
    std::cout << std::endl;
}

int main() {
    test_network_automata();
}
