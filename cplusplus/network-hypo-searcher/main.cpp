#include "hypo-searcher.h"

#include <iostream>

using namespace NNetworkHypoSearcher;

void test_hypo_searcher() {
    HypoSearcher searcher("/home/stepan/git-repos/typos-corrector/python/model/dataset/",
                          "/home/stepan/git-repos/typos-corrector/python/model/parameters/",
                          "/home/stepan/git-repos/typos-corrector/python/model/first-mistake-statistics");
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
    test_hypo_searcher();
}
