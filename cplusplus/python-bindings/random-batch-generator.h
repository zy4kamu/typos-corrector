#pragma once

#include <random>
#include <string>
#include <vector>

#include "compressor.h"
#include "contaminator.h"
#include "dataset.h"

class RandomBatchGenerator {
public:
    RandomBatchGenerator(const DataSet& dataset, const Contaminator& contaminator, const Compressor& compressor);
    void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size);
private:
    std::string get_random_string();

    const DataSet&      dataset;
    const Contaminator& contaminator;
    const Compressor&   compressor;
    std::mt19937        generator;
};
