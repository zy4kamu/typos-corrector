#pragma once

#include <random>
#include <string>

class Contaminator {
public:
    Contaminator(double mistake_probability);
    std::string swap_random_chars(const std::string& token) const;
    std::string replace_random_char(const std::string& token) const;
    std::string add_random_char(const std::string& token) const;
    std::string remove_random_char(const std::string& token) const;
    std::string contaminate(const std::string& token) const;
private:
    char get_random_qwerty_neighbour(char letter, bool allow_repeat) const;
    char get_random_char() const;

    const double mistake_probability;
    mutable std::mt19937 generator;
};

