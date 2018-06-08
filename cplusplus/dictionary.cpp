#include "dictionary.h"
#include "utils.h"

#include <boost/filesystem.hpp>
#include <fstream>

Dictionary::Dictionary(const std::string& input_file) {
    if (!boost::filesystem::exists(input_file)) {
        throw std::runtime_error("absent input file " + input_file);
    }
    std::ifstream reader(input_file);
    std::string token;
    while (getline(reader, token)) {
        dictionary.insert(std::move(token));
    }
}

Dictionary::Dictionary(const std::string& input_file, size_t min_chars, size_t max_chars) {
    if (!boost::filesystem::exists(input_file)) {
        throw std::runtime_error("absent input file " + input_file);
    }
    std::ifstream reader(input_file);
    std::string token;
    while (getline(reader, token)) {
        if (token.size() >= min_chars && token.size() <= max_chars) {
            dictionary.insert(std::move(token));
        }
    }
}

const std::unordered_set<std::string>& Dictionary::get() const {
    return dictionary;
}

