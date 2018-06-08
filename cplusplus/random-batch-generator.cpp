#include "random-batch-generator.h"
#include "dictionary.h"
#include "utils.h"

#include <algorithm>
#include <boost/make_unique.hpp>
#include <iostream>
#include <memory>

RandomBatchGenerator::RandomBatchGenerator(const char* input_file, double mistake_probability)
    : Contaminator(mistake_probability), generator(1) {
    std::cout << input_file << std::endl;
    Dictionary dictionary(input_file);
    for (const std::string& item : dictionary.get()) {
        tokens.push_back(item);
        weights.push_back(1);
    }
    distribution = std::discrete_distribution<size_t>(weights.begin(), weights.end());
}

void RandomBatchGenerator::generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch,
                           size_t message_size, size_t batch_size) {
    std::fill(clean_batch, clean_batch + message_size * batch_size, Z_INT - A_INT + 1);
    std::fill(contaminated_batch, contaminated_batch + message_size * batch_size, Z_INT - A_INT + 1);
    for (size_t i = 0; i < batch_size; ) {
        std::string clean_token = tokens[distribution(generator)];
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

std::unique_ptr<RandomBatchGenerator> DATASET_PREPARER;

extern "C" {

void create_random_batch_generator(const char* prefix_tree_file, double mistake_probability) {
    DATASET_PREPARER = boost::make_unique<RandomBatchGenerator>(prefix_tree_file,
                                                                mistake_probability);
}

void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch,
                           size_t message_size, size_t batch_size, double ) {
    DATASET_PREPARER->generate_random_batch(clean_batch, contaminated_batch,
                                            message_size, batch_size);
}

} // extern "C"
