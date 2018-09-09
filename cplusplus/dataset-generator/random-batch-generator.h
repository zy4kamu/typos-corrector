#pragma once

#include <random>
#include <string>
#include <vector>

#include "contaminator.h"
#include "../dataset/compressor.h"
#include "../dataset/dataset.h"

class RandomBatchGenerator {
public:
    RandomBatchGenerator(const DataSet& dataset, const Contaminator& contaminator, const Compressor& compressor);
    void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size);
private:
    std::vector<std::string> generate_one_example();
    std::string get_clean_string(const std::vector<std::string>& example) const;

    const DataSet&      dataset;
    const Contaminator& contaminator;
    const Compressor&   compressor;
    std::mt19937        generator;

    std::discrete_distribution<size_t> country_distribution  { 0.05, 0.95 };
    std::discrete_distribution<size_t> city_distribution     { 0.2, 0.8 };
    std::discrete_distribution<size_t> district_distribution { 0.1, 0.9 };
    std::discrete_distribution<size_t> index_distribution    { 0.05, 0.95 };
    std::discrete_distribution<size_t> street_distribution   { 0.9, 0.1 };
    std::discrete_distribution<size_t> house_distribution    { 0.3, 0.7 };
};

