#include "hypo-searcher.h"

#include <chrono>
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

        auto start = std::chrono::steady_clock::now();
        std::vector<std::string> hypos = searcher.search(input);
        auto end = std::chrono::steady_clock::now();
        for (const std::string& hypo : hypos) {
            std::cout << hypo << std::endl;
        }
        std::cout << "spent " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ...\n" << std::endl;
    }
}

void test_dataset_generator() {
    std::mt19937 generator(1);
    DataSet dataset("/home/stepan/datasets/europe-hierarchy");
    for (size_t i = 0; i < 100; ++i) {
        std::vector<const DataSet::Entity*> entities = dataset.get_random_item(generator);
        for (const DataSet::Entity* entity : entities) {
            std::cout << entity->type << " " << entity->name << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // test_dataset_generator();
    test_hypo_searcher(argc, argv);
}
