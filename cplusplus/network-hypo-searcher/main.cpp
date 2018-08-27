#include "compressed-lstm.h"
#include "network-automata.h"
#include "hypo-searcher.h"
#include "opencl-connector.h"
#include "../utils/utils.h"
#include "../python-bindings/dataset.h"

#include <cblas.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>

#include "common.h"

using namespace NNetworkHypoSearcher;

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
    for (const std::string& token : {"masterdamweg", "ahrlemweg"}) {
        automata.encode_message(token, probabilities);
        std::vector<float_type> first_char_probabilities = probabilities;

        // first pass
        for (size_t i = 0; i < 25; ++i) {
            size_t index = argmax(probabilities);
            char ch = to_char(static_cast<int32_t>(index));
            std::cout << ch;
            automata.apply(ch, probabilities);
        }
        std::cout << std::endl;

        // second pass
        automata.reset();
        probabilities = first_char_probabilities;
        for (size_t i = 0; i < 25; ++i) {
            size_t index = argmax(probabilities);
            char ch = to_char(static_cast<int32_t>(index));
            std::cout << ch;
            automata.apply(ch, probabilities);
        }
        std::cout << std::endl;
    }
}

void test_hypo_searcher() {
    HypoSearcher searcher("/home/stepan/git-repos/typos-corrector/python/model/update-regions/",
                          "/home/stepan/git-repos/typos-corrector/python/model/parameters/",
                          "/home/stepan/git-repos/typos-corrector/python/model/first-mistake-statistics");

    /*
    std::vector<std::string> hypos = searcher.search("bosenlomerweg");
    for (const std::string& hypo : hypos) {
        std::cout << hypo << std::endl;
    }
    */

    std::string input;
    while (true) {
        std::cout << "Input something: ";
        std::getline(std::cin, input);

        std::vector<std::string> hypos = searcher.search(input);
        for (const std::string& hypo : hypos) {
            std::cout << hypo << std::endl;
        }
    }
}

int main() {
    DataSet dataset("/home/stepan/git-repos/typos-corrector/python/model/dataset");
    std::mt19937 generator;
    for (size_t i = 0; i < 100; ++i) {
        std::tuple<std::string, std::string, std::string> data = dataset.get_random_item(generator);
        std::cout << std::get<0>(data) <<  " " << std::get<1>(data) << " " << std::get<2>(data) << std::endl;
    }
    test_hypo_searcher();
}
