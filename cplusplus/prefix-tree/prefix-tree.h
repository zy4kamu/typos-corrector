#pragma once

#include <string>
#include <vector>

class PrefixTree {
public:
    PrefixTree(const std::string& filename);
    ~PrefixTree();
    bool check(const std::string& message);
 private:
    int file_desrciptor;
    size_t file_size;
    char* data;
};
