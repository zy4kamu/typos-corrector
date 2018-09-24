#include "vw-model.h"

#include <algorithm>
#include <iostream>

int main() {
    NVWModel::VWModel model("/home/stepan/country-dataset/model");
    std::string input;
    while (true) {
        std::cout << "Input something: ";
        std::getline(std::cin, input);
        const std::vector<float>& predictions = model.predict(input);
        std::vector<size_t> indexes;
        for (size_t i = 0; i < predictions.size(); ++i) {
            indexes.push_back(i);
        }
        std::sort(indexes.begin(), indexes.end(), [&predictions](size_t i, size_t j) { return predictions[i] > predictions[j]; });
        for (size_t i : indexes) {
            std::cout << model.label(i) << ": " << predictions[i] << std::endl;
        }
        std::cout << "\n" << std::endl;
    }
}
