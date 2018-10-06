#include "i-database-requester.h"
#include "utils.h"

#include <limits>
#include <sstream>

namespace NNetworkHypoSearcher {

namespace {

void append(std::string& where_append, const std::string& to_append, char separator) {
    if (where_append.empty()) {
        where_append = to_append;
    } else {
        where_append += separator;
        where_append += to_append;
    }
}

void append_length(size_t& where_append, size_t to_append) {
    if (where_append == 0) {
        where_append = to_append;
    }
    else {
        where_append += to_append + 1;
    }
}

} // anonymous namespace

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

size_t IDataBaseRequester::find_max_prefix_one_entity(const std::string& entity, size_t limit,
                                                      std::vector<std::string>& pretendents) const {
    if (find_entities_present_in_database(entity, limit, pretendents)) {
        return entity.length();
    }
    size_t start = 0;
    size_t end = entity.length();
    while (start + 1 < end) {
        size_t middle = (start + end) / 2;
        std::vector<std::string> temp_pretendents;
        if (find_entities_present_in_database(entity.substr(0, middle), limit, temp_pretendents)) {
            pretendents = std::move(temp_pretendents);
            start = middle;
        } else {
            end = middle;
        }
    }
    return start;
}

size_t IDataBaseRequester::find_max_prefix_full_query(const std::string& query, char separator) const {
    size_t prefix_length = 0;
    std::stringstream reader(query);
    std::string current_entity;
    while (getline(reader, current_entity, separator)) {
        if (contains_digit(current_entity)) {
            append_length(prefix_length, current_entity.length());
        } else {
            size_t one_token_max_prefix_length = find_max_prefix_one_entity(current_entity);
            append_length(prefix_length, one_token_max_prefix_length);
            if (one_token_max_prefix_length != current_entity.length()) {
                break;
            }
        }
    }
    return prefix_length;
}

size_t IDataBaseRequester::levenstein_request(const std::string& query, size_t request_limit, size_t levenstein_limit,
                                              char separator, std::string& corrected_query) const {
    std::stringstream reader(query);
    size_t prefix_length = 0;
    bool calculate_prefix = true;
    std::string current_entity;

    while (getline(reader, current_entity, separator)) {
        // if entity contains digit we classify it as house number or index and skip it
        if (contains_digit(current_entity)) {
            append(corrected_query, current_entity, separator);
            if (calculate_prefix) {
                append_length(prefix_length, current_entity.length());
            }
            continue;
        }

        // make request to database
        std::vector<std::string> pretendents;
        size_t one_token_max_prefix_length = find_max_prefix_one_entity(current_entity, request_limit, pretendents);

        // update prefix length
        if (calculate_prefix) {
            append_length(prefix_length, one_token_max_prefix_length);
            calculate_prefix = one_token_max_prefix_length == current_entity.length();
        }

        // if there is a full prefix match just exist: everything is already fine
        if (reader.tellg() == -1 && calculate_prefix) {
            corrected_query = query;
            break;
        }

        // if there is a full prefix match after some levenstein corrections: append entity and continue
        if (reader.tellg() == -1 && one_token_max_prefix_length == current_entity.length()) {
            append(corrected_query, current_entity, separator);
            break;
        }

        // if there are too many pretendents and there is no perfect prefix match, break: we cannot predict anything
        if (pretendents.empty()) {
            corrected_query.clear();
            break;
        }

        // iterate over pretendents and find the nearest levenstein
        size_t min_levenstein = std::numeric_limits<size_t>::max();
        const std::string* best_pretendent = nullptr;
        for (const std::string& pretendent : pretendents) {
            size_t distance = levenstein_distance(current_entity, pretendent); // TODO: allow prefixes
            if (distance < min_levenstein) {
                min_levenstein = distance;
                best_pretendent = &pretendent;
            }
        }

        // adopt leventein hypo
        if (min_levenstein <= levenstein_limit) {
            levenstein_limit -= min_levenstein;
            if (!corrected_query.empty()) {
                corrected_query += separator;
            }
            corrected_query += *best_pretendent;
        } else {
            corrected_query.clear();
            break;
        }
    }

    return prefix_length;
}

} // namespace NNetworkHypoSearcher
