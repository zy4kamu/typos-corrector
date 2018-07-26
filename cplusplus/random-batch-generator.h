#pragma once

#include <random>
#include <string>
#include <vector>

#include "compressor.h"
#include "contaminator.h"
#include "update-regions.h"

class RandomBatchGenerator {
public:
    RandomBatchGenerator(const UpdateRegionSet& update_region_set, const Contaminator& contaminator,
                         const Compressor& compressor);
    int32_t generate_random_batch_on_one_ur(int32_t* clean_batch, int32_t* contaminated_batch, size_t message_size, size_t batch_size);
    void generate_random_batch_on_all_urs(int32_t* clean_batch, int32_t* contaminated_batch, size_t message_size, size_t batch_size);
private:
    size_t get_random_update_region();
    const std::string& get_random_token(size_t update_region_id);

    const UpdateRegionSet& update_region_set;
    const Contaminator&    contaminator;
    const Compressor&      compressor;
    std::mt19937           generator;

    std::discrete_distribution<size_t> distribution_over_update_regions;
    std::vector<std::uniform_int_distribution<size_t>> distributions_inside_update_regions;
};
