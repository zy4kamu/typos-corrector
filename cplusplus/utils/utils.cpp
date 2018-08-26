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

size_t levenstein_distance(const std::string& first, const std::string& second) {
    if (first.empty()) {
        return second.length();
    }
    if (second.empty()) {
        return first.length();
    }
    size_t first_grid_size = first.length() + 1;
    size_t second_grid_size = second.length() + 1;
    std::vector<size_t> grid(first_grid_size * second_grid_size, 0);
    for (size_t i = 0; i < first_grid_size; ++i) {
        grid[first_grid_size * second.length() + i] = first.length() - i;
    }
    for (size_t i = 0; i < second_grid_size; ++i) {
        grid[first.length() + i * first_grid_size] = second.length() - i;
    }
    for (size_t i = second.length() - 1; i + 1 != 0; --i) {
        for (size_t j = first.length() - 1; j + 1 != 0; --j) {
            if (second[i] == first[j]) {
                grid[i * first_grid_size + j] = grid[(i + 1) * first_grid_size + (j + 1)];
            } else {
                grid[i * first_grid_size + j] = 1 + std::min(grid[(i + 1) * first_grid_size + (j + 1)],
                        std::min(grid[(i + 1) * first_grid_size + j], grid[i * first_grid_size + (j + 1)]));
            }
        }
    }
    return grid[0];
}

size_t levenstein_distance(const char* first, const char* second, size_t message_size) {
  if (message_size == 0) {
    return 0;
  }
  size_t grid_size = message_size + 1;
  std::vector<size_t> grid(grid_size * grid_size, 0);
  for (size_t i = 0; i < grid_size; ++i) {
    grid[grid_size * message_size + i] = grid[message_size + i * grid_size] = grid_size - i - 1;
  }
  for (size_t i = message_size - 1; i + 1 != 0; --i) {
    for (size_t j = message_size - 1; j + 1 != 0; --j) {
      if (first[i] == second[j]) {
        grid[i * grid_size + j] = grid[(i + 1) * grid_size + (j + 1)];
      } else {
        grid[i * grid_size + j] = 1 + std::min(grid[(i + 1) * grid_size + (j + 1)],
                                               std::min(grid[(i + 1) * grid_size + j], grid[i * grid_size + (j + 1)]));
      }
    }
  }
  return grid[0];
}
