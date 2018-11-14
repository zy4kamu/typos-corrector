#pragma once

#include <cstring>
#include <map>
#include <string>
#include <vector>

struct PrefixTreeBuilderNode {
   std::map<char, PrefixTreeBuilderNode> content;

   PrefixTreeBuilderNode& operator[](char letter);
   std::vector<char> to_string() const;
};

class PrefixTreeBuilder {
public:
    void add(const std::string& message);
    std::vector<char> to_string() const;
private:
    PrefixTreeBuilderNode root;
};
