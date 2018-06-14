#pragma once

#include <vector>
#include <string>
#include <random>

#include "contaminator.h"

struct RandomBatchGenerator: private Contaminator {
    RandomBatchGenerator(const char* input_file, double mistake_probability);
    void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch,
                               size_t message_size, size_t batch_size);
private:
    std::vector<std::string> tokens;
    std::vector<size_t> weights;
    std::discrete_distribution<size_t> token_distribution;
    std::uniform_int_distribution<size_t> prefix_distribution;
    std::mt19937 generator;
};


