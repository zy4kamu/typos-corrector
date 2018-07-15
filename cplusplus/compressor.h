#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class UpdateRegionSet;

class Compressor {
public:
    Compressor(const UpdateRegionSet& update_region_set);
    std::string compress(const std::string& decompressed) const;
    const std::vector<std::string>& decompress(const std::string& compressed) const;
private:
    std::unordered_map<std::string, std::vector<std::string>> decompress_map;
};
