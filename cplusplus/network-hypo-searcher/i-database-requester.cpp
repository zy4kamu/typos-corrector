#include "i-database-requester.h"

namespace NNetworkHypoSearcher {

size_t IDataBaseRequester::find_max_prefix_one_token(const std::string& token) const {
    if (is_present_in_database(token)) {
        return token.length();
    }
    size_t start = 0;
    size_t end = token.length();
    while (start + 1 < end) {
        size_t middle = (start + end) / 2;
        if (is_present_in_database(token.substr(0, middle))) {
            start = middle;
        } else {
            end = middle;
        }
    }
    return start;
}

} // namespace NNetworkHypoSearcher
