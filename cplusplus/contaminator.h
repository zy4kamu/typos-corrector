#pragma once

#include <random>
#include <string>

class Contaminator {
public:
    Contaminator(double mistake_probability);
    std::string swap_random_chars(const std::string& token);
    std::string replace_random_char(const std::string& token);
    std::string add_random_char(const std::string& token);
    std::string remove_random_char(const std::string& token);
    std::string contaminate(const std::string& token);
private:
    char get_random_qwerty_neighbour(char letter, bool allow_repeat);
    char get_random_char();

    double mistake_probability;
    std::mt19937 generator;
};

