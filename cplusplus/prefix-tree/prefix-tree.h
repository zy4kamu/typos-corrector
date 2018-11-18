#pragma once

#include <bitset>
#include <string>
#include <vector>

class PrefixTreeState;
class PrefixTreeMaster;

class PrefixTreeState {
public:
    void move(char letter);
    PrefixTreeState move(char letter) const;
    const std::vector<char>& get_transitions() const;
private:
    void reset_pass(const char* pointer);

    const char* current_pointer;
    std::vector<char> transitions;

    friend class PrefixTreeMaster;
};

class PrefixTreeMaster {
public:
    PrefixTreeMaster(const std::string& filename);
    ~PrefixTreeMaster();
    const PrefixTreeState& get_initial_state();
private:
    void reset_pass(const char* pointer);
    int file_desrciptor;
    size_t file_size;
    const char* root;

    PrefixTreeState initial_state;
};
