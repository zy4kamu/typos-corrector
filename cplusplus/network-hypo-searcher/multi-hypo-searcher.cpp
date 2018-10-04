#include "multi-hypo-searcher.h"

#include <fstream>
#include <map>
#include <sys/types.h>
#include <sys/stat.h>
#include <unordered_set>

#include <boost/make_unique.hpp>

// 1. TODO: implement states in hypo-searcher:
// ConsumedQuery <---> ReadyForNewQuery
// LoadedData <---> UnloadedData
// 2. TODO: next action must be generated on the fly
// 3. TODO: why it can't find oosterdokstraat ???
// 4. TODO: refactor vw-model and move it inside this project (not sure)
// 5. TODO: train VW model with typos

namespace NNetworkHypoSearcher {

namespace {

int directory_exists(const std::string& folder) {
    struct stat info;
    if (stat(folder.c_str(), &info) != 0) {
        return 0;
    } else if (info.st_mode & S_IFDIR) {
        return 1;
    } else {
        return 0;
    }
}

const float_type PROBABILITY_TO_MAKE_MISTAKE = 0.2;

} // anonymous namespace

MultiHypoSearcher::MultiHypoSearcher(const std::string& typos_corrector_folder,
                                     const std::vector<std::string>& countries,
                                     const std::string& vw_model_file):
    vw_model(vw_model_file), current_hypo_searcher(nullptr), countries(countries), current_country_index(std::string::npos) {
    for (const std::string& country : countries) {
        size_t country_index = country_to_index.size();
        country_to_index[country] = country_index;
        const std::string lstm_folder = typos_corrector_folder + "/" + country;
        if (directory_exists(lstm_folder)) {
            country_to_searcher[country_index] = boost::make_unique<HypoSearcher>(lstm_folder);
        }
    }
}

void MultiHypoSearcher::initialize(const std::string& initial_query, size_t num_attempts) {
    this->initial_query = initial_query;
    commands = {{ CommandType::Initialize, std::string::npos }};
    NVWModel::VWModel::MapType predictions = vw_model.predict(initial_query, country_to_index);
    std::unordered_set<size_t> processed_countries;
    std::unordered_set<size_t> initialized_countries;
    for (size_t i = 0; i < num_attempts; ++i) {
        auto iter = predictions.begin();
        const size_t country_index = iter->second;
        float_type weight = iter->first;
        if (processed_countries.find(country_index) == processed_countries.end()) {
            commands.emplace_back(Command { CommandType::SearchUncorrected, country_index });
            processed_countries.insert(country_index);
            predictions.erase(iter);
            predictions.insert(std::make_pair(weight * PROBABILITY_TO_MAKE_MISTAKE, country_index));
            continue;
        }
        if (initialized_countries.find(country_index) == initialized_countries.end()) {
            commands.emplace_back(Command { CommandType::InitializeCorrector, country_index });
            initialized_countries.insert(country_index);
        }
        commands.emplace_back(Command { CommandType::SearchCorrected, country_index });
        predictions.erase(iter);
        predictions.insert(std::make_pair(weight * country_to_searcher[country_index]->get_probability_not_to_correct(), country_index));
        continue;
    }
    commands_iterator = commands.begin();
}

bool MultiHypoSearcher::next(std::string& country, std::string& hypo) {
    while (commands_iterator != commands.end()) {
        ++commands_iterator;
        size_t country_index = commands_iterator->country_index;
        country = countries[country_index];
        CommandType type = commands_iterator->type;
        switch (type) {
            case CommandType::Initialize:
                throw std::runtime_error("Shouldn't be called in current implementation");
            case CommandType::SearchUncorrected:
                current_query = hypo = initial_query;
                return true;
            case CommandType::InitializeCorrector:
                if (current_hypo_searcher != nullptr && country_index != current_country_index) {
                    current_hypo_searcher->unload();
                }
                current_country_index = country_index;
                current_hypo_searcher = country_to_searcher[country_index].get();
                current_hypo_searcher->load();
                current_hypo_searcher->initialize(initial_query);
            break;
            case CommandType::SearchCorrected:
                current_query = hypo = country_to_searcher[country_index]->generate_next_hypo();
                return true;
        }
    }
    return false;
}

bool MultiHypoSearcher::check(IDataBaseRequester& requester) {
    switch(commands_iterator->type) {
    case CommandType::SearchCorrected:
        return current_hypo_searcher->check_hypo_in_database(requester);
    default:
        return requester.find_max_prefix_full_query(current_query) == current_query.length();
    }
}

} // namespace NNetworkHypoSearcher
