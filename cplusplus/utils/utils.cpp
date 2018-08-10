#include "utils.h"

#include <cassert>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>

const int32_t A_INT = static_cast<int32_t>('a');
const int32_t Z_INT = static_cast<int32_t>('z');
const size_t NUM_LETTERS = Z_INT - A_INT + 2;

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
        ch = static_cast<char>(std::tolower(ch));
        if (acceptable(ch)) {
            cleaned += ch;
        }
    }
    return cleaned;
}


template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

size_t get_file_size(const char* filename) {
    assert(filename != nullptr);
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    } else {
        exit(EXIT_FAILURE);
    }
}

size_t get_file_size(const std::string& filename) {
    return get_file_size(filename.c_str());
}
