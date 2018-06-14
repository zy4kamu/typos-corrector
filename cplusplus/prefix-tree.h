#include <string>
#include <utility>
#include <vector>

using String = std::basic_string<unsigned char>;

class PrefixTree {
public:
    static void Create(const std::string& input_file, const std::string& output_file);
    PrefixTree(const std::string& input_file);
    size_t match(const std::string& token) const;
    void viterbi(const double* logits, size_t length, size_t num_hypos, std::vector<std::string>& output_tokens,
                 std::vector<double>& output_logits) const;
private:
    String content;
};
