#include "multi-hypo-searcher.h"

#include <chrono>
#include <iostream>
#include <string>

using namespace NNetworkHypoSearcher;

namespace {

const size_t MAX_PASS = 20;
const std::string INPUT_FOLDER = "/home/stepan/git-repos/typos-corrector/python/model/";

struct DataSetRequester : private DataSet, public IDataBaseRequester {
    DataSetRequester(const std::string& input_folder): DataSet(input_folder, std::string::npos, false) {
    }

    bool is_present_in_database(const std::string& token) const override {
        return !DataSet::find_by_prefix(token, 1).empty();
    }
};

} // anonymous namespace

void test_hypo_searcher() {
    DataSetRequester requester(INPUT_FOLDER + "dataset/all");
    HypoSearcher searcher(INPUT_FOLDER + "good-models/north-97.93/");
    std::string input;
    while (true) {
        std::cout << "Input something: ";
        std::getline(std::cin, input);

        auto start = std::chrono::steady_clock::now();
        searcher.initialize(input);
        for (size_t i = 0; i < 20; ++i) {
            const std::string& hypo = searcher.generate_next_hypo();
            std::cout << hypo << std::endl;
            bool found_full_match = searcher.check_hypo_in_database(requester);
            if (found_full_match) {
                std::cout << "found :-)" << std::endl;
                break;
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << "spent " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ...\n" << std::endl;
    }
}

void test_multi_hypo_searcher() {
    DataSetRequester requester(INPUT_FOLDER + "dataset/all");
    MultiHypoSearcher searcher({ INPUT_FOLDER + "/good-models/north-97.93/",
                                 INPUT_FOLDER + "/good-models/slavic+english-97.6/",
                                 INPUT_FOLDER + "/good-models/south-98.3/" },
                               "/home/stepan/country-dataset/model");
    std::string input;
    while (true) {
        std::cout << "Input something: ";
        std::getline(std::cin, input);

        auto start = std::chrono::steady_clock::now();
        const std::string country = searcher.initialize(input);
        std::cout << "predicted country: " << country << std::endl;
        for (size_t i = 0; i < 20; ++i) {
            const std::string& hypo = searcher.generate_next_hypo();
            std::cout << hypo << std::endl;
            bool found_full_match = searcher.check_hypo_in_database(requester);
            if (found_full_match) {
                std::cout << "found :-)" << std::endl;
                break;
            }
        }
        auto end = std::chrono::steady_clock::now();
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
    // test_hypo_searcher();
    test_multi_hypo_searcher();
}
