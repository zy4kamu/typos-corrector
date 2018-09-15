#pragma once

#include <string>

namespace NNetworkHypoSearcher {

class IDataBaseRequester {
public:
    virtual bool is_present_in_database(const std::string& token) const = 0;
    size_t find_max_prefix_one_token(const std::string& token) const;
};

} // namespace NNetworkHypoSearcher
