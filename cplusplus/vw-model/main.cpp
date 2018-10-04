#include "vw-model.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

int main() {
    // Broken: doesn't work
    /*
    const std::string home = getenv("HOME");
    NVWModel::VWModel model(home + "/git-repos/typos-corrector/country-dataset/model");
    std::string input;
    while (true) {
        std::cout << "Input something: ";
        std::getline(std::cin, input);
        auto predictions = model.predict(input);
        for (const auto& item : predictions) {
            std::cout << item.first << " " << item.second << std::endl;
        }
        std::cout << "\n" << std::endl;
    }
    */
}
