#include "contaminator.h"
#include "utils.h"

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
    std::uniform_int_distribution<size_t> char_distribution(A_INT, Z_INT);
    char letter = char_distribution(generator);

    std::uniform_int_distribution<size_t> distribution(0, token.size() - 1);
    size_t index = distribution(generator);
    if (index == token.size()) {
        return token + letter;
    }
    return token.substr(0, index) + letter + token.substr(index + 1);
}

std::string Contaminator::add_random_char(const std::string& token) {
    std::uniform_int_distribution<size_t> char_distribution(A_INT, Z_INT + 1);
    char letter = char_distribution(generator);
    if (static_cast<int32_t>(letter) == Z_INT + 1) {
        letter = ' ';
    }
    std::uniform_int_distribution<size_t> distribution(0, token.size());
    size_t index = distribution(generator);
    return token.substr(0, index) + letter + token.substr(index);
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
