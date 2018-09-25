#include "vw-model.h"

#include <algorithm>
#include <iostream>

int main() {
    NVWModel::VWModel model("/home/stepan/country-dataset/model");
    std::string input;
    while (true) {
        std::cout << "Input something: ";
        std::getline(std::cin, input);
        std::vector<std::pair<std::string, NVWModel::float_type>> predictions = model.predict(input);
        for (const auto& item : predictions) {
            std::cout << item.first << " " << item.second << std::endl;
        }
        std::cout << "\n" << std::endl;
    }
}
