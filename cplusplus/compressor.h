#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class Compressor {
public:
    Compressor() = default;
    Compressor(const std::string& file);
    Compressor(const std::vector<std::string>& tokens);
    void set(const std::vector<std::string>& tokens);
    std::string compress(const std::string& decompressed);
    const std::vector<std::string>& decompress(const std::string& compressed) const;
private:
    std::unordered_map<std::string, std::vector<std::string>> decompress_map;
};

