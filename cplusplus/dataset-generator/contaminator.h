#pragma once

#include <random>
#include <string>

#include "ngrams.h"

class Contaminator {
public:
    Contaminator(const std::string& ngrams_file, double mistake_probability);
    std::string contaminate(const std::vector<std::string>& example, size_t message_size) const;
private:
    char get_random_qwerty_neighbour(char letter, bool allow_repeat) const;
    char get_random_char() const;
    char get_random_char(const std::string& message, size_t index) const;

    std::string swap_random_chars(const std::string& message) const;
    std::string replace_random_char(const std::string& message) const;
    std::string add_random_char(const std::string& message) const;
    std::string remove_random_char(const std::string& message) const;

    std::string contaminate(const std::string& message) const;

    Ngrams ngrams;
    const double mistake_probability;
    mutable std::mt19937 generator;
};

