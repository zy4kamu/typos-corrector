#include "multi-hypo-searcher.h"

#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include <boost/make_unique.hpp>

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

} // anonymous namespace

MultiHypoSearcher::MultiHypoSearcher(const std::string& typos_corrector_folder,
                                     const std::vector<std::string>& countries,
                                     const std::string& vw_model_file):
    vw_model(vw_model_file) {
    for (const std::string& country : countries) {
        const std::string lstm_folder = typos_corrector_folder + "/" + country;
        if (directory_exists(lstm_folder)) {
            country_to_searcher[country] = boost::make_unique<HypoSearcher>(lstm_folder);
        }
    }
}

void MultiHypoSearcher::initialize(const std::string& input, size_t /* num_attempts */) {
    this->initial_query = input;
    commands = {
        { CommandType::Initialize,          ""                },
        { CommandType::SearchUncorrected,   "the netherlands" },
        { CommandType::InitializeCorrector, "the netherlands" },
        { CommandType::SearchCorrected,     "the netherlands" },
        { CommandType::SearchCorrected,     "the netherlands" },
        { CommandType::SearchUncorrected,   "united kingdom"  },
        { CommandType::InitializeCorrector, "united kingdom"  },
        { CommandType::SearchCorrected,     "united kingdom"  },
        { CommandType::SearchCorrected,     "united kingdom"  }
    };
    commands_iterator = commands.begin();
}

bool MultiHypoSearcher::next(std::string& country, std::string& hypo) {
    while (commands_iterator != commands.end()) {
        ++commands_iterator;
        country = commands_iterator->country;
        CommandType type = commands_iterator->type;
        switch (type) {
            case CommandType::Initialize:
                throw std::runtime_error("Shouldn't be called in current implementation");
            case CommandType::SearchUncorrected:
                current_query = hypo = initial_query;
                return true;
            case CommandType::InitializeCorrector:
                country_to_searcher[country]->initialize(initial_query);
            break;
            case CommandType::SearchCorrected:
                current_query = hypo = country_to_searcher[country]->generate_next_hypo();
                return true;
        }
    }
    return false;
}

bool MultiHypoSearcher::check(IDataBaseRequester& requester) {
    switch(commands_iterator->type) {
    case CommandType::SearchCorrected:
        return country_to_searcher[commands_iterator->country]->check_hypo_in_database(requester);
    default:
        return requester.find_max_prefix_full_query(current_query) == current_query.length();
    }
}

} // namespace NNetworkHypoSearcher
