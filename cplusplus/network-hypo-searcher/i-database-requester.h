#pragma once

#include <string>
#include <vector>

namespace NNetworkHypoSearcher {

class IDataBaseRequester {
public:
    // IMPORTANT: we suppose there is a match if there is prefix match, in full query the policy is the same
    virtual bool is_one_entity_present_in_database(const std::string& entity) const = 0;
    virtual bool find_entities_present_in_database(const std::string& entity, size_t limit, std::vector<std::string>& pretendents) const = 0;


    size_t find_max_prefix_one_entity(const std::string& entity) const;
    size_t find_max_prefix_one_entity(const std::string& entity, size_t limit, std::vector<std::string>& pretendents) const;
    size_t find_max_prefix_full_query(const std::string& query, char separator = '|') const;
    size_t levenstein_request(const std::string& query, size_t request_limit, size_t levenstein_limit, char separator,
                              std::string& corrected_query) const;
};

} // namespace NNetworkHypoSearcher
