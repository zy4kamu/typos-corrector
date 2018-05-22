#include <memory>
#include <string>
#include <unordered_map>

class PrefixTree {
public:
    PrefixTree();
    PrefixTree(const std::string& file_name);
    size_t match(const std::string& token) const;
    size_t match(const char* token) const;
    size_t match(const char* token, size_t length) const;
    void add(const std::string& token);
    void add(const char* token, size_t length);
    void save(const std::string& file_name) const;
private:
    struct Node {
        std::unordered_map<char, std::unique_ptr<Node>> transitions;
    };

    void write_node(std::ofstream& writer, const Node& node) const;
    std::unique_ptr<Node> read_node(std::ifstream& reader) const;

    std::unique_ptr<Node> root;
};
