#pragma once

#include <bitset>
#include <string>
#include <vector>

class PrefixTree {
public:
    PrefixTree(const std::string& filename);
    ~PrefixTree();
    const std::vector<char>& get_transitions() const;
    void reset_pass();
    void move(char letter);
    bool check(const std::string& message);
private:
    void reset_pass(const char* pointer);
    int file_desrciptor;
    size_t file_size;
    const char* root;

    const char* current_pointer;
    std::vector<char> transitions;
};
