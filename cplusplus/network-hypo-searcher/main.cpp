#include "hypo-searcher.h"

#include <iostream>
#include <string>

using namespace NNetworkHypoSearcher;

void test_hypo_searcher(int argc, char* argv[]) {
    std::string input_folder;
    if (argc > 1) {
        input_folder = argv[1];
        input_folder += "/";
    }
    HypoSearcher searcher(input_folder + "dataset/",
                          input_folder + "parameters/",
                          input_folder + "first-mistake-statistics");
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

int main(int argc, char* argv[]) {
    test_hypo_searcher(argc, argv);
}
