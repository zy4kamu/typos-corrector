#include "random-batch-generator.h"
#include "utils.h"

#include <algorithm>
#include <boost/make_unique.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

namespace {
const size_t MIN_PREFIX_SIZE = 4;
const bool LEARN_BY_PREFIXES = false;
} // anonymous namespace

RandomBatchGenerator::RandomBatchGenerator(const char* input_file, double mistake_probability)
    : Contaminator(mistake_probability), generator(1) {
    std::cout << input_file << std::endl;
    std::ifstream reader(input_file);
    std::string token;
    while (getline(reader, token)) {
        tokens.push_back(token);
        weights.push_back(1);
    }
    token_distribution = std::discrete_distribution<size_t>(weights.begin(), weights.end());
    prefix_distribution = std::uniform_int_distribution<size_t>(0, std::numeric_limits<size_t>::max());
}

void RandomBatchGenerator::generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch,
                           size_t message_size, size_t batch_size) {
    std::fill(clean_batch, clean_batch + message_size * batch_size, Z_INT - A_INT + 1);
    std::fill(contaminated_batch, contaminated_batch + message_size * batch_size, Z_INT - A_INT + 1);
    for (size_t i = 0; i < batch_size; ) {
        std::string clean_token = tokens[token_distribution(generator)];
        if (LEARN_BY_PREFIXES && clean_token.length() > MIN_PREFIX_SIZE) {
            size_t prefix_size = MIN_PREFIX_SIZE + prefix_distribution(generator) % (clean_token.length() - MIN_PREFIX_SIZE);
            clean_token.resize(prefix_size);
        }
        std::string contaminated_token = contaminate(clean_token);
        if (clean_token.size() > message_size || contaminated_token.size() > message_size)
        {
            continue;
        }
        int32_t shift = static_cast<int32_t>(i * message_size);
        std::transform(clean_token.begin(), clean_token.end(), clean_batch + shift, to_int);
        std::transform(contaminated_token.begin(), contaminated_token.end(), contaminated_batch + shift, to_int);
        ++i;
    }
}

/****** PYTHON EXPORTS ******/

std::unique_ptr<RandomBatchGenerator> DATASET_GENERATOR;

extern "C" {

void create_random_batch_generator(const char* prefix_tree_file, double mistake_probability) {
    DATASET_GENERATOR = boost::make_unique<RandomBatchGenerator>(prefix_tree_file,
                                                                mistake_probability);
}

void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch,
                           size_t message_size, size_t batch_size, double ) {
    DATASET_GENERATOR->generate_random_batch(clean_batch, contaminated_batch,
                                            message_size, batch_size);
}

} // extern "C"
