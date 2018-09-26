#include "random-batch-generator.h"

#include "../network-hypo-searcher/hypo-searcher.h"
#include "../network-hypo-searcher/i-database-requester.h"

#include <iostream>

const size_t MESSAGE_SIZE = 25;

struct DataSetRequester : public NNetworkHypoSearcher::IDataBaseRequester {
    DataSetRequester(const DataSet& dataset): dataset(dataset) {
    }

    bool is_present_in_database(const std::string& token) const override {
        return !dataset.find_by_prefix(token, 1).empty();
    }
private:
    const DataSet& dataset;
};

int main(int argc, char* argv[]) {
    const std::string input_folder = "/home/stepan/git-repos/typos-corrector/python/model/";
    DataSet dataset(input_folder + "dataset/slavic");
    Contaminator contaminator(input_folder + "ngrams", 0.2);
    RandomBatchGenerator batch_generator(dataset, contaminator, MESSAGE_SIZE);
    DataSetRequester requester(dataset);
    NNetworkHypoSearcher::HypoSearcher searcher(input_folder + "parameters/",
                                                input_folder + "first-mistake-statistics");

    std::string real, contaminated;
    size_t num_found = 0;
    for (size_t i = 0; i < 10000; ++i) {
        batch_generator.next(real, contaminated);
        searcher.initialize(contaminated);
        for (size_t j = 0; j < 20; ++j) {
            searcher.generate_next_hypo();
            bool found_full_match = searcher.check_hypo_in_database(requester);
            if (found_full_match) {
                ++num_found;
                break;
            }
        }
        std::cout << num_found << " of " << 10000 << "; accuracy="
                  << static_cast<double>(num_found) / static_cast<double>(i + 1) << std::endl;
    }
}
