#include "contaminator.h"
#include "utils.h"

#include <unordered_map>

namespace {

std::unordered_map<char, std::vector<char>> QWERTY_MAP =
{
    {'q', {'a', 'w', 's'}},
    {'w', {'q', 'a', 's', 'd', 'e'}},
    {'e', {'w', 's', 'd', 'f', 'r'}},
    {'r', {'e', 'd', 'f', 'g', 't'}},
    {'t', {'r', 'f', 'g', 'h', 'y'}},
    {'y', {'t', 'g', 'h', 'j', 'u'}},
    {'u', {'y', 'h', 'j', 'k', 'i'}},
    {'i', {'u', 'j', 'k', 'l', 'o'}},
    {'o', {'i', 'k', 'l', 'p'}},
    {'p', {'o', 'l'}},
    {'a', {'q', 'w', 's', 'x', 'z'}},
    {'s', {'q', 'w', 'e', 'd', 'x', 'z', 'a'}},
    {'d', {'w', 'e', 'r', 'f', 'c', 'x', 's'}},
    {'f', {'e', 'r', 't', 'g', 'v', 'c', 'd'}},
    {'g', {'r', 't', 'y', 'h', 'b', 'v', 'f'}},
    {'h', {'t', 'y', 'u', 'j', 'n', 'b', 'g'}},
    {'j', {'h', 'y', 'u', 'i', 'k', 'm', 'n'}},
    {'k', {'j', 'u', 'i', 'o', 'l', 'm'}},
    {'l', {'k', 'i', 'o', 'p'}},
    {'z', {'a', 's', 'x'}},
    {'x', {'z', 'a', 's', 'd', 'c'}},
    {'c', {'x', 's', 'd', 'f', 'v'}},
    {'v', {'c', 'd', 'f', 'g', 'b'}},
    {'b', {'v', 'f', 'g', 'h', 'n'}},
    {'n', {'b', 'g', 'h', 'j', 'm'}},
    {'m', {'n', 'h', 'j', 'k'}},
    {' ', {'z', 'x', 'c', 'v', 'b', 'n', 'm'}}
};

} // anonymous namespace

Contaminator::Contaminator(double mistake_probability): mistake_probability(mistake_probability)
                                        , generator(1) {
}

std::string Contaminator::swap_random_chars(const std::string& token) {
    if (token.size() < 2) {
        return token;
    }
    std::uniform_int_distribution<size_t> distribution(0, token.size() - 2);
    size_t index = distribution(generator);
    return token.substr(0, index) + token[index + 1] + token[index] + token.substr(index + 2);
}

std::string Contaminator::replace_random_char(const std::string& token) {
    if (token.empty()) {
        return token;
    }
    std::uniform_int_distribution<size_t> distribution(0, token.size() - 1);
    size_t index = distribution(generator);
    return token.substr(0, index) + get_random_qwerty_neighbour(token[index], false) + token.substr(index + 1);
}

std::string Contaminator::add_random_char(const std::string& token) {
    if (token.empty()) {
        return std::string(1, get_random_char());
    }
    std::uniform_int_distribution<size_t> distribution(0, token.size());
    size_t index = distribution(generator);
    if (index == token.size()) {
        return token + get_random_qwerty_neighbour(token.back(), true);
    }
    return token.substr(0, index) + get_random_qwerty_neighbour(token[index], true) + token.substr(index);
}

std::string Contaminator::remove_random_char(const std::string& token) {
    std::uniform_int_distribution<size_t> distribution(0, token.size() - 1);
    size_t index = distribution(generator);
    return token.substr(0, index) + token.substr(index + 1);
}

std::string Contaminator::contaminate(const std::string& token) {
    std::bernoulli_distribution contaminate_distribution(mistake_probability);
    std::uniform_int_distribution<int> distribution(0, 3);
    std::string contaminated_token = token;
    for (size_t i = 0; i < token.size(); ++i) {
        if (contaminate_distribution(generator)) {
            int type = distribution(generator);
            switch (type) {
                case 0:
                    contaminated_token = swap_random_chars(contaminated_token);
                    break;
                case 1:
                    contaminated_token = replace_random_char(contaminated_token);
                    break;
                case 2:
                    contaminated_token = add_random_char(contaminated_token);
                    break;
                case 3:
                    contaminated_token = remove_random_char(contaminated_token);
                    break;
            }
        }
    }
    return contaminated_token;
}

// helpers

char Contaminator::get_random_qwerty_neighbour(char letter, bool allow_repeat) {
    const std::vector<char>& neighbours = QWERTY_MAP.at(letter);
    if (allow_repeat) {
        std::uniform_int_distribution<size_t> distribution(0, neighbours.size());
        size_t index = distribution(generator);
        return index == neighbours.size() ? letter : neighbours[index];
    }
    std::uniform_int_distribution<size_t> distribution(0, neighbours.size() - 1);
    size_t index = distribution(generator);
    return neighbours[index];
}

char Contaminator::get_random_char() {
    std::uniform_int_distribution<size_t> char_distribution(A_INT, Z_INT + 1);
    char letter = char_distribution(generator);
    if (static_cast<int32_t>(letter) == Z_INT + 1) {
        letter = ' ';
    }
    return letter;
}


