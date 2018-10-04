#include "ngrams.h"
#include "../network-hypo-searcher/utils.h"

#include <cassert>
#include <fstream>

Ngrams::Ngrams(const std::string& input_file): ngram_size(0) {
    // read from file
    std::ifstream reader(input_file);
    std::string line;
    while (getline(reader, line)) {
        std::vector<std::string> splitted = split(line, '|');
        assert(splitted.size() == 3);
        assert(splitted[1].length() == 1);
        const std::string& ngram = splitted[0];
        assert(ngram_size == 0 || ngram_size == ngram.length());
        ngram_size = ngram.length();
        transition_probs[ngram].resize(EFFECTIVE_NUM_LETTERS, 0);
        const char next_char = splitted[1][0];
        const double counter = std::stod(splitted[2]);
        transition_probs[ngram][to_int(next_char)] = counter;
    }

    // normalize
    for (auto& kvp : transition_probs) {
        double sum = 1;
        for (double count : kvp.second) {
            sum += count;
        }
        for (double& prob : kvp.second) {
            prob /= sum;
        }
    }
}

const std::vector<double>& Ngrams::get_probabities(const std::string& ngram) const {
    assert(ngram.length() == ngram_size);
    auto found = transition_probs.find(ngram);
    if (found != transition_probs.end()) {
        return found->second;
    }
    static const std::vector<double> empty_vector;
    return empty_vector;
}
