#include "i-database-requester.h"
#include "../utils/utils.h"

#include <sstream>

namespace NNetworkHypoSearcher {

size_t IDataBaseRequester::find_max_prefix_one_entity(const std::string& entity) const {
    if (is_one_entity_present_in_database(entity)) {
        return entity.length();
    }
    size_t start = 0;
    size_t end = entity.length();
    while (start + 1 < end) {
        size_t middle = (start + end) / 2;
        if (is_one_entity_present_in_database(entity.substr(0, middle))) {
            start = middle;
        } else {
            end = middle;
        }
    }
    return start;
}

size_t IDataBaseRequester::find_max_prefix_full_query(const std::string& query) const {
    size_t prefix_length = 0;
    std::stringstream reader(query);
    std::string current_entity;
    while (getline(reader, current_entity, '|')) {
        if (contains_digit(current_entity)) {
            prefix_length += prefix_length > 0 ? current_entity.length() + 1 : current_entity.length();
        } else {
            size_t one_token_max_prefix_length = find_max_prefix_one_entity(current_entity);
            prefix_length += prefix_length > 0 ? one_token_max_prefix_length + 1 : one_token_max_prefix_length;
            if (one_token_max_prefix_length != current_entity.length()) {
                break;
            }
        }
    }
    return prefix_length;
}

} // namespace NNetworkHypoSearcher
