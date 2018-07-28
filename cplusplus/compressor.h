#pragma once

#include <string>
#include <map>
#include <vector>

class UpdateRegionSet;

class Compressor {
public:
    Compressor(const UpdateRegionSet& update_region_set, size_t message_size);
    std::string compress(const std::string& decompressed) const;
    const std::vector<std::string>& decompress(const std::string& compressed) const;
    std::vector<std::string> find_by_prefix(const std::string& prefix, size_t max_number) const;
    size_t get_message_size() const;
private:
    std::map<std::string, std::vector<std::string>> decompress_map;
    size_t message_size;
};
