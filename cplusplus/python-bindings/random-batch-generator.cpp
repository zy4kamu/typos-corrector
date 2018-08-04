#include "random-batch-generator.h"

#include "compressor.h"
#include "utils.h"

#include <algorithm>
// #include <iostream>

RandomBatchGenerator::RandomBatchGenerator(const UpdateRegionSet& update_region_set, const Contaminator& contaminator,
                                           const Compressor& compressor)
    : update_region_set(update_region_set)
    , contaminator(contaminator)
    , compressor(compressor)
    , generator(1) {
    const std::vector<UpdateRegion>& update_regions = update_region_set.update_regions;
    std::vector<size_t> update_region_sizes(update_regions.size());
    std::transform(update_regions.begin(), update_regions.end(), update_region_sizes.begin(),
                   [](const UpdateRegion& update_region) { return update_region.tokens.size(); });
    distribution_over_update_regions = std::discrete_distribution<size_t>(update_region_sizes.begin(),
                                                                          update_region_sizes.end());
    std::transform(update_regions.begin(), update_regions.end(), std::back_inserter(distributions_inside_update_regions),
                   [](const UpdateRegion& update_region) {
                          return std::uniform_int_distribution<size_t>(0, update_region.tokens.size() - 1);
                   });
}

size_t RandomBatchGenerator::get_random_update_region() {
    return distribution_over_update_regions(generator);
}

const std::string& RandomBatchGenerator::get_random_token(size_t update_region_id) {
    size_t token_index = distributions_inside_update_regions[update_region_id](generator);
    return update_region_set.update_regions[update_region_id].tokens[token_index];
}

int32_t RandomBatchGenerator::generate_random_batch_on_one_ur(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size) {
    size_t message_size = compressor.get_message_size();
    std::fill(clean_batch, clean_batch + message_size * batch_size, Z_INT - A_INT + 1);
    std::fill(contaminated_batch, contaminated_batch + message_size * batch_size, Z_INT - A_INT + 1);
    size_t update_region_id = get_random_update_region();
    for (size_t i = 0; i < batch_size; ++i) {
        const std::string& clean_token = get_random_token(update_region_id);
        std::string compressed_clean_token = compressor.compress(clean_token);
        std::string contaminated_token = contaminator.contaminate(clean_token);
        if (contaminated_token.length() > message_size) {
            contaminated_token = contaminated_token.substr(0, message_size);
        }
        int32_t shift = static_cast<int32_t>(i * message_size);
        std::transform(compressed_clean_token.begin(), compressed_clean_token.end(), clean_batch + shift, to_int);
        std::transform(contaminated_token.begin(), contaminated_token.end(), contaminated_batch + shift, to_int);
    }
    return update_region_id;
}

void RandomBatchGenerator::generate_random_batch_on_all_urs(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size) {
    size_t message_size = compressor.get_message_size();
    std::fill(clean_batch, clean_batch + message_size * batch_size, Z_INT - A_INT + 1);
    std::fill(contaminated_batch, contaminated_batch + message_size * batch_size, Z_INT - A_INT + 1);
    for (size_t i = 0; i < batch_size; ++i) {
        size_t update_region_id = get_random_update_region();
        const std::string clean_token = get_random_token(update_region_id);
        std::string compressed_clean_token = compressor.compress(clean_token);
        std::string contaminated_token = contaminator.contaminate(clean_token);
        if (contaminated_token.length() > message_size) {
            contaminated_token = contaminated_token.substr(0, message_size);
        }
        int32_t shift = static_cast<int32_t>(i * message_size);
        // std::cout << compressed_clean_token << " " << contaminated_token << std::endl;
        std::transform(compressed_clean_token.begin(), compressed_clean_token.end(), clean_batch + shift, to_int);
        std::transform(contaminated_token.begin(), contaminated_token.end(), contaminated_batch + shift, to_int);
    }
}
