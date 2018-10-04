#include "contaminator.h"
#include "../network-hypo-searcher/utils.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>

namespace {

const std::unordered_map<char, std::vector<char>> QWERTY_MAP = {
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

const std::vector<char> ALHABET = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z'
};

} // anonymous namespace

Contaminator::Contaminator(const std::string& ngrams_file, double mistake_probability)
    : ngrams(ngrams_file), mistake_probability(mistake_probability), generator(1) {
}

std::string Contaminator::swap_random_chars(const std::string& message) const {
    if (message.size() < 2) {
        return message;
    }
    std::uniform_int_distribution<size_t> distribution(0, message.size() - 2);
    size_t index = distribution(generator);
    return message.substr(0, index) + message[index + 1] + message[index] + message.substr(index + 2);
}

std::string Contaminator::replace_random_char(const std::string& message) const {
    if (static_cast<size_t>(std::count(message.begin(), message.end(), '|')) == message.length()) {
        return message;
    }
    std::uniform_int_distribution<size_t> distribution(0, message.size() - 1);
    while (true) {
        size_t index = distribution(generator);
        if (message[index] == '|') {
            continue;
        }
        return message.substr(0, index) + get_random_char(message, index) + message.substr(index + 1);
    }
}

std::string Contaminator::add_random_char(const std::string& message) const {
    if (message.empty()) {
        return std::string(1, get_random_char());
    }
    std::uniform_int_distribution<size_t> distribution(0, message.size());
    size_t index = distribution(generator);
    return message.substr(0, index) + get_random_char(message, index) + message.substr(index);
}

std::string Contaminator::remove_random_char(const std::string& message) const {
    if (static_cast<size_t>(std::count(message.begin(), message.end(), '|')) == message.length()) {
        return message;
    }
    std::uniform_int_distribution<size_t> distribution(0, message.size() - 1);
    while (true) {
        size_t index = distribution(generator);
        if (message[index] == '|') {
            continue;
        }
        return message.substr(0, index) + message.substr(index + 1);
    }
}

std::string Contaminator::contaminate(const std::vector<std::string>& example, size_t message_size) const {
    std::string message;
    std::vector<std::string> digit_parts;
    for (const std::string& component : example) {
        if (!message.empty()) {
            message += " ";
        }
        if (contains_digit(component)) {
            digit_parts.push_back(component);
            message += "|";
        } else {
            message += component;
        }
    }
    std::string contaminated_string = contaminate(message);
    for (const std::string& digit_part : digit_parts) {
        size_t placeholder_position = contaminated_string.find('|');
        assert(placeholder_position != std::string::npos);
        contaminated_string = contaminated_string.substr(0, placeholder_position) + digit_part +
                contaminated_string.substr(placeholder_position + 1);
    }
    if (contaminated_string.length() > message_size) {
        contaminated_string = contaminated_string.substr(0, message_size);
    }
    return contaminated_string;
}

std::string Contaminator::contaminate(const std::string& message) const {
    size_t mistake_counter = 0;
    std::bernoulli_distribution contaminate_distribution(mistake_probability);
    std::uniform_int_distribution<int> distribution(0, 3);
    std::string contaminated_string = message;
    for (size_t i = 0; i < message.size(); ++i) {
        if (contaminate_distribution(generator)) {
            if (++mistake_counter == 3) { // no more than 3 mistakes per message
                break;
            }
            int type = distribution(generator);
            switch (type) {
                case 0:
                    contaminated_string = swap_random_chars(contaminated_string);
                    break;
                case 1:
                    contaminated_string = replace_random_char(contaminated_string);
                    break;
                case 2:
                    contaminated_string = add_random_char(contaminated_string);
                    break;
                case 3:
                    contaminated_string = remove_random_char(contaminated_string);
                    break;
            }
        }
    }
    return contaminated_string;
}

// helpers

char Contaminator::get_random_char(const std::string& message, size_t index) const {
    std::discrete_distribution<int> type_distribution({ 0.45, 0.45, 0.1});
    int type = type_distribution(generator);
    switch(type) {
    case 0:
    {
        // ngrams
        std::string prefix;
        if (index < ngrams.size()) {
            prefix = std::string(ngrams.size() - index, ' ');
            prefix += message.substr(0, index);
        } else {
            prefix = message.substr(index - ngrams.size(), ngrams.size());
        }
        const std::vector<double>& probs = ngrams.get_probabities(prefix);
        if (!probs.empty()) {
            std::discrete_distribution<int> char_distribution(probs.begin(), probs.end());
            int index = char_distribution(generator);
            return to_char(index);
        }
    }
    case 1:
        // qwerty
        return get_random_qwerty_neighbour(index == message.length() ? message.back() : message[index], true);
        // random
    case 2:
        return get_random_char();
    }
    throw std::runtime_error("get_random_char error");
}

char Contaminator::get_random_qwerty_neighbour(char letter, bool allow_repeat) const {
    auto found = QWERTY_MAP.find(letter);
    if (found == QWERTY_MAP.end()) {
        return get_random_char();
    }
    const std::vector<char>& neighbours = found->second;
    if (allow_repeat) {
        std::uniform_int_distribution<size_t> distribution(0, neighbours.size());
        size_t index = distribution(generator);
        return index == neighbours.size() ? letter : neighbours[index];
    }
    std::uniform_int_distribution<size_t> distribution(0, neighbours.size() - 1);
    size_t index = distribution(generator);
    return neighbours[index];
}

char Contaminator::get_random_char() const {
    std::uniform_int_distribution<size_t> char_distribution(A_INT, Z_INT + 1);
    char letter = char_distribution(generator);
    if (static_cast<int32_t>(letter) == Z_INT + 1) {
        letter = ' ';
    }
    return letter;
}
