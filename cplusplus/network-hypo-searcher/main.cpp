#include "multi-hypo-searcher.h"

#include <chrono>
#include <iostream>
#include <string>

using namespace NNetworkHypoSearcher;

namespace {

const size_t MAX_PASS = 20;
const std::string INPUT_FOLDER = "/home/stepan/git-repos/typos-corrector/";

struct DataSetRequester : private DataSet, public IDataBaseRequester {
    DataSetRequester(const std::string& input_folder): DataSet(input_folder, std::string::npos, false) {
    }

    bool is_present_in_database(const std::string& token) const override {
        return !DataSet::find_by_prefix(token, 1).empty();
    }
};

double elapsed_time(std::chrono::time_point<std::chrono::steady_clock> start,
                    std::chrono::time_point<std::chrono::steady_clock> end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000000;
}

} // anonymous namespace

void test_hypo_searcher(const std::string& country) {
    auto start = std::chrono::steady_clock::now();
    DataSetRequester requester(INPUT_FOLDER + "dataset/by-country/" + country);
    auto end = std::chrono::steady_clock::now();
    std::cout << "downloaded dataset in " << elapsed_time(start, end) << " seconds" << std::endl;

    start = std::chrono::steady_clock::now();
    HypoSearcher searcher(INPUT_FOLDER + "python/models/binaries/" + country);
    end = std::chrono::steady_clock::now();
    std::cout << "downloaded model in " << elapsed_time(start, end) << " seconds" << std::endl;

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
        std::cout << "spent " << elapsed_time(start, end) << " seconds ...\n" << std::endl;
    }
}

// DOESN'T WORK, BROKEN
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

int main(int argc, char* argv[]) {
    test_hypo_searcher("italy");
    // test_multi_hypo_searcher();
}
