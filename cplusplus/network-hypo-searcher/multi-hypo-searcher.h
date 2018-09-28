#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "hypo-searcher.h"
#include "multi-hypo-searcher.h"
#include "../vw-model/vw-model.h"

namespace NNetworkHypoSearcher {

class MultiHypoSearcher {
public:
    MultiHypoSearcher(const std::vector<std::string>& input_folders, const std::string& vw_model_file);
    std::string initialize(const std::string& input);
    const std::string& generate_next_hypo();
    bool check_hypo_in_database(IDataBaseRequester& requester);
private:
    NVWModel::VWModel vw_model;
    std::vector<HypoSearcher> hypo_searchers;
    std::unordered_map<std::string, size_t> country_to_searcher_index;

    HypoSearcher* current_searcher;
};

} // namespace NNetworkHypoSearcher
