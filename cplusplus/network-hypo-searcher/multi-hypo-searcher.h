#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "hypo-searcher.h"
#include "multi-hypo-searcher.h"
#include "../vw-model/vw-model.h"

namespace NNetworkHypoSearcher {

enum class CommandType {
    Initialize,
    SearchUncorrected,
    InitializeCorrector,
    SearchCorrected
};

struct Command {
    CommandType type;
    std::string country;
};

class MultiHypoSearcher {
public:
    MultiHypoSearcher(const std::string& typos_corrector_folder,
                      const std::vector<std::string>& countries,
                      const std::string& vw_model_file);
    void initialize(const std::string& initial_query, size_t num_attempts);
    bool next(std::string& country, std::string& hypo);
    bool check(IDataBaseRequester& requester);
private:
    NVWModel::VWModel vw_model;
    std::unordered_map<std::string, std::unique_ptr<HypoSearcher>> country_to_searcher;
    std::vector<Command> commands;
    std::vector<Command>::const_iterator commands_iterator;

    std::string initial_query;
    std::string current_query;
};

} // namespace NNetworkHypoSearcher
