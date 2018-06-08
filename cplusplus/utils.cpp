#include "utils.h"

#include <cassert>

const int32_t A_INT = static_cast<int32_t>('a');
const int32_t Z_INT = static_cast<int32_t>('z');

bool acceptable(char ch) {
    return ch == ' ' || ('a' <= ch && ch <= 'z');
}

int32_t to_int(char ch) {
    assert(acceptable(ch));
    if (ch == ' ') {
       return Z_INT - A_INT + 1;
    }
    return static_cast<int32_t>(ch) - A_INT;
}

char to_char(int32_t number) {
    assert(0 <= number && number <= Z_INT - A_INT + 1);
    return number == Z_INT - A_INT + 1 ? ' ' : static_cast<char>(A_INT + number);
}

std::string clean_token(const std::string& token) {
    std::string cleaned;
    for (char ch: token) {
        ch = std::tolower(ch);
        if (acceptable(ch)) {
            cleaned += ch;
        }
    }
    return cleaned;
}
