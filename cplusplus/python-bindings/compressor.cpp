#include "compressor.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>


#include <boost/algorithm/string/predicate.hpp>
#include <boost/make_unique.hpp>

Compressor::Compressor(const DataSet& dataset, size_t message_size)
    : message_size(message_size) {
    for (const std::string& country : dataset.get_countries()) {
        decompress_map[compress(country)].push_back(country);
    }
    for (const DataSet::CitiesStreets& cities_streets : dataset.get_cities_streets()) {
        for (const std::string& city : cities_streets.keys) {
            decompress_map[compress(city)].push_back(city);
        }
        for (const DataSet::Streets& streets : cities_streets.values) {
            for (const std::string& street : streets.values) {
                decompress_map[compress(street)].push_back(street);
            }
        }
    }
}

std::string Compressor::compress(const std::string& token) const {
    std::string compressed(token);
    std::replace(compressed.begin(), compressed.end(), 'c', 'k');
    std::replace(compressed.begin(), compressed.end(), 'w', 'v');

    if (compressed.empty()) {
        return compressed;
    }
    std::string to_return(1, compressed[0]);
    for (size_t i = 1; i < compressed.size(); ++i) {
        if (compressed[i] != compressed[i - 1]) {
            to_return += compressed[i];
        }
    }
    if (to_return.length() > message_size) {
        to_return = to_return.substr(0, message_size);
    }
    return to_return;
}

const std::vector<std::string>& Compressor::decompress(const std::string& compressed) const {
    auto found = decompress_map.find(compressed);
    static const std::vector<std::string> empty_vector;
    return found == decompress_map.end() ? empty_vector : found->second;
}

std::vector<std::string> Compressor::find_by_prefix(const std::string& prefix, size_t max_number) const {
    std::vector<std::string> results;
    auto found = decompress_map.lower_bound(prefix);
    while (found != decompress_map.end() && boost::starts_with(found->first, prefix)) {
        results.insert(results.end(), found->second.begin(), found->second.end());
        if (results.size() > max_number) {
            return results;
        }
        ++found;
    }
    return results;
}

size_t Compressor::get_message_size() const {
    return message_size;
}
