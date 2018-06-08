#pragma once

#include <string>
#include <unordered_set>

class Dictionary {
public:
    explicit Dictionary(const std::string& input_file);
    Dictionary(const std::string& input_file, size_t min_chars, size_t max_chars);
    const std::unordered_set<std::string>& get() const;
private:
    std::unordered_set<std::string> dictionary;
};

