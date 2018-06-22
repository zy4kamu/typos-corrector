#include "utils.h"

#include <cassert>
#include <vector>

#include <string>
#include <iostream>

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

extern "C" {

size_t levenstein(const char* first, const char* second, size_t message_size) {
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

} // extern C
