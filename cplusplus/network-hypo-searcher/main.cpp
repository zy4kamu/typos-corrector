#include "multi-hypo-searcher.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace NNetworkHypoSearcher;

namespace {

const size_t MAX_PASS = 20;
const std::string home = getenv("HOME");
const std::string INPUT_FOLDER = home + "/git-repos/typos-corrector/";

struct DataSetRequester : private DataSet, public IDataBaseRequester {
    DataSetRequester(const std::string& input_folder): DataSet(input_folder, false) {
    }

    bool is_one_entity_present_in_database(const std::string& token) const override {
        return !DataSet::find_by_prefix(token, 1).empty();
    }

    bool find_entities_present_in_database(const std::string& entity, size_t limit, std::vector<std::string>& pretendents) const override {
        pretendents = DataSet::find_by_prefix(entity, limit + 1);
        if (pretendents.size() > limit) {
            pretendents.clear();
            return true;
        }
        return !pretendents.empty();
    }
};

double elapsed_time(std::chrono::time_point<std::chrono::steady_clock> start,
                    std::chrono::time_point<std::chrono::steady_clock> end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000000;
}

} // anonymous namespace

void test_prefix_tree_hypo_searcher(const std::string& country) {
    PrefixTreeMaster prefix_tree(INPUT_FOLDER + "dataset/by-country/" + country + "/prefix-tree");
    HypoSearcher searcher(INPUT_FOLDER + "python/models/binaries/" + country);
    searcher.load();

    std::string input;
    while (true) {
        std::cout << "Input something: ";
        std::getline(std::cin, input);
        std::vector<std::string> hypos = searcher.cover_probability(input, 0.99, 15, prefix_tree);
        for (const std::string& hypo : hypos) {
            std::cout << hypo << std::endl;
        }
    }
}

void test_hypo_searcher(const std::string& country) {
    auto start = std::chrono::steady_clock::now();
    DataSetRequester requester(INPUT_FOLDER + "dataset/by-country/" + country);
    auto end = std::chrono::steady_clock::now();
    std::cout << "downloaded dataset in " << elapsed_time(start, end) << " seconds" << std::endl;

    start = std::chrono::steady_clock::now();
    HypoSearcher searcher(INPUT_FOLDER + "python/models/binaries/" + country);
    searcher.load();
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
            std::string levenstein_correction;
            bool found_full_match = searcher.check_hypo_in_database(requester, levenstein_correction);
            if (found_full_match) {
                std::cout << "found: " << levenstein_correction << ":-)" << std::endl;
                break;
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << "spent " << elapsed_time(start, end) << " seconds ...\n" << std::endl;
    }
}

void test_multi_hypo_searcher() {
    auto start = std::chrono::steady_clock::now();
    DataSetRequester requester(INPUT_FOLDER + "dataset/all/");
    auto end = std::chrono::steady_clock::now();
    std::cout << "downloaded dataset in " << elapsed_time(start, end) << " seconds" << std::endl;

    start = std::chrono::steady_clock::now();
    MultiHypoSearcher searcher(INPUT_FOLDER + "python/models/binaries/",
                               { "the netherlands", "united kingdom", "italy" },
                               home + "/git-repos/typos-corrector/country-dataset/model");
    end = std::chrono::steady_clock::now();
    std::cout << "downloaded model in " << elapsed_time(start, end) << " seconds" << std::endl;

    const size_t num_attempts = 40;
    std::string input, country, hypo;
    while (true) {
        std::cout << "Input something: ";
        std::getline(std::cin, input);

        auto start = std::chrono::steady_clock::now();
        searcher.initialize(input, num_attempts);
        for (size_t i = 0; i < num_attempts; ++i) {
            searcher.next(country, hypo);
            std::cout << country << ": " << hypo << std::endl;
            std::string levenstein_correction;
            bool found_full_match = searcher.check(requester, levenstein_correction);
            if (found_full_match) {
                std::cout << "found: \033[1;31m" << levenstein_correction << "\033[0m in \033[1;31m" << country << "\033[0m :-)\n";
                break;
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << "spent " << elapsed_time(start, end) << " seconds ...\n" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    test_prefix_tree_hypo_searcher("denmark");
    // test_hypo_searcher("france");
    // test_multi_hypo_searcher();
}
