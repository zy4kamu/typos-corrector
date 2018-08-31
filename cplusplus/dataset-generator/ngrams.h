#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class Ngrams {
public:
    Ngrams(const std::string& input_file);
    size_t size() const { return ngram_size; }
    const std::vector<double>& get_probabities(const std::string& ngram) const;
private:
    std::unordered_map<std::string, std::vector<double>> transition_probs;
    size_t ngram_size;
};
