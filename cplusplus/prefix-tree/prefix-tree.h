#pragma once

#include <string>
#include <vector>

struct PrefixTreeNode {
    PrefixTreeNode(const char* data = nullptr);
    std::vector<char> transitions;
    PrefixTreeNode move(char letter);
    const char* data;
};

class PrefixTree {
public:
    PrefixTree(const std::string& filename);
    const char* get_root() const { return root.data; }
    ~PrefixTree();
    bool check(const std::string& message);
    bool get(const std::string& message, size_t limit, std::vector<std::string>& pretendents);
 private:
    const char* forward(const std::string& message) const;
    bool walk(const char* pointer, const std::string& prefix, size_t limit, std::vector<std::string>& pretendents);

    int file_desrciptor;
    size_t file_size;
    PrefixTreeNode root;
};
