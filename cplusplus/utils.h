#pragma once

#include <string>
#include <vector>

extern const int32_t A_INT;
extern const int32_t Z_INT;
extern const size_t  NUM_LETTERS;

bool acceptable(char ch);
int32_t to_int(char ch);
char to_char(int32_t number);
std::string clean_token(const std::string& token);
std::vector<std::string> split(const std::string &s, char delim);

extern "C" {

size_t levenstein(const char* first, const char* second, size_t message_size);

} // extern C
