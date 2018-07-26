#include "compressor.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>


#include <boost/algorithm/string/predicate.hpp>
#include <boost/make_unique.hpp>

#include "update-regions.h"

Compressor::Compressor(const UpdateRegionSet& update_region_set) {
    for (const UpdateRegion& update_region : update_region_set.update_regions) {
        for (const std::string& token : update_region.tokens) {
            decompress_map[compress(token)].push_back(token);
        }
    }
    size_t num_collisions = 0;
    size_t num_total = 0;
    for (const auto& item : decompress_map) {
        size_t current = item.second.size();
        num_total += current;
        if (current > 1) {
            num_collisions += current;
        }
    }
    std::cout << "compressor: " << num_collisions << " collisions of " << num_total
              << ": " << static_cast<double>(num_collisions) / static_cast<double>(num_total) << std::endl;
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
    return to_return;
}

const std::vector<std::string>& Compressor::decompress(const std::string& compressed) const {
    auto found = decompress_map.find(compressed);
    static const std::vector<std::string> empty_vector;
    return found == decompress_map.end() ? empty_vector : found->second;
}

std::vector<std::string> Compressor::find_by_prefix(const std::string& prefix, size_t max_number) const {
    std::vector<std::string> results;
    auto found = decompress_map.find(prefix);
    while (found != decompress_map.end() && boost::starts_with(found->first, prefix)) {
        results.insert(results.end(), found->second.begin(), found->second.end());
        if (results.size() > max_number) {
            return results;
        }
    }
    return results;
}
