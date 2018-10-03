#pragma once

#include <string>

namespace NNetworkHypoSearcher {

class IDataBaseRequester {
public:
    virtual bool is_one_entity_present_in_database(const std::string& token) const = 0;
    size_t find_max_prefix_one_entity(const std::string& entity) const;
    size_t find_max_prefix_full_query(const std::string& query) const;
};

} // namespace NNetworkHypoSearcher
