#pragma once

#include <string>
#include <vector>

extern const int32_t A_INT;
extern const int32_t Z_INT;
extern const int32_t SPACE_INT;
extern const int32_t SEPARATOR_INT;
extern const size_t  EFFECTIVE_NUM_LETTERS;
extern const size_t  NUM_LETTERS;

bool acceptable(char ch);
int32_t to_int(char ch);
char to_char(int32_t number);
std::string clean_token(const std::string& token);
std::vector<std::string> split(const std::string &s, char delim);
bool contains_digit(const std::string& token);

size_t get_file_size(const char *filename);
size_t get_file_size(const std::string& filename);
size_t levenstein_distance(const char* first, const char* second, size_t message_size);
size_t levenstein_distance(const std::string& first, const std::string& second);
std::vector<std::string> read_directory(const std::string& name);

#define _unused(x) ((void)(x))
