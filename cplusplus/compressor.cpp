#include "compressor.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>

#include <boost/make_unique.hpp>


Compressor::Compressor(const std::vector<std::string>& tokens) {
    set(tokens);
}

Compressor::Compressor(const std::string& file) {
    std::vector<std::string> tokens;
    std::ifstream reader(file);
    std::string token;
    while (getline(reader, token)) {
        tokens.push_back(std::move(token));
    }
    set(tokens);
}

void Compressor::set(const std::vector<std::string>& tokens) {
    for (const std::string& token : tokens) {
        decompress_map[compress(token)].push_back(token);
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

std::string Compressor::compress(const std::string& token) {
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

/****** PYTHON EXPORTS ******/

std::unique_ptr<Compressor> COMPRESSOR;

extern "C" {

void create_compressor(const char* input_file) {
    COMPRESSOR = boost::make_unique<Compressor>(input_file);
}

void decompress(const char* token, char* output) {
    const std::vector<std::string>& decompressed = COMPRESSOR->decompress(token);
    std::stringstream stream;
    for (const std::string& decompresed_token : decompressed) {
        stream << decompresed_token << "|";
    }
    const std::string concatenated = stream.str();
    if (!concatenated.empty()) {
        std::memcpy(output, concatenated.c_str(), concatenated.length() - 1);
    }
}

} // extern "C"
