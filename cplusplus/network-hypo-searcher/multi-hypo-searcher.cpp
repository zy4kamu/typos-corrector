#include "multi-hypo-searcher.h"

#include <fstream>

namespace NNetworkHypoSearcher {

MultiHypoSearcher::MultiHypoSearcher(const std::vector<std::string>& input_folders,
                                     const std::string& vw_model_file):
    vw_model(vw_model_file), current_searcher(nullptr) {
    std::string country;
    for (const std::string& input_folder : input_folders) {
        std::ifstream reader(input_folder + "/countries");
        while (getline(reader, country)) {
            country_to_searcher_index[country] = hypo_searchers.size();
        }
        hypo_searchers.emplace_back(input_folder);
    }
}

std::string MultiHypoSearcher::initialize(const std::string& input) {
    std::vector<std::pair<std::string, float_type>> probabilities = vw_model.predict(input);
    size_t index = country_to_searcher_index[probabilities[0].first];
    current_searcher = &hypo_searchers[index];
    current_searcher->initialize(input);
    return probabilities[0].first;
}

const std::string& MultiHypoSearcher::generate_next_hypo() {
    return current_searcher->generate_next_hypo();
}

bool MultiHypoSearcher::check_hypo_in_database(IDataBaseRequester& requester) {
    return current_searcher->check_hypo_in_database(requester);
}

} // namespace NNetworkHypoSearcher
