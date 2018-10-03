#include "random-batch-generator.h"

#include "../network-hypo-searcher/hypo-searcher.h"
#include "../network-hypo-searcher/i-database-requester.h"

#include <iostream>

// SOUTH:  95.64%
// SLAVIC: 97.26%
// NORTH:  97.86%

const size_t MESSAGE_SIZE = 25;

struct DataSetRequester : public NNetworkHypoSearcher::IDataBaseRequester {
    DataSetRequester(const DataSet& dataset): dataset(dataset) {
    }

    bool is_one_entity_present_in_database(const std::string& token) const override {
        return !dataset.find_by_prefix(token, 1).empty();
    }
private:
    const DataSet& dataset;
};

// DOESN'T WORK, BROKEN
void check_accuracy() {
    const std::string input_folder = "/home/stepan/git-repos/typos-corrector/python/model/";
    DataSet dataset(input_folder + "dataset/slavic");
    Contaminator contaminator(input_folder + "ngrams", 0.2);
    RandomBatchGenerator batch_generator(dataset, contaminator, MESSAGE_SIZE);
    DataSetRequester requester(dataset);
    NNetworkHypoSearcher::HypoSearcher searcher(input_folder + "parameters/");

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

void generate_country_dataset() {
    std::mt19937 generator(1);
    DataSet dataset("/home/stepan/git-repos/typos-corrector/python/model/dataset/north", 0);
    Contaminator contaminator("/home/stepan/git-repos/typos-corrector/python/model/ngrams", 0.2);
    RandomBatchGenerator batch_generator(dataset, contaminator, MESSAGE_SIZE);
    batch_generator.generate_country_dataset("/home/stepan/git-repos/typos-corrector/country-dataset/dataset/", 1000, 1000);
}


int main(int argc, char* argv[]) {
    // check_accuracy();
    generate_country_dataset();
}
