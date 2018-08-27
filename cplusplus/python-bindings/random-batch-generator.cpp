#include "random-batch-generator.h"

#include "compressor.h"
#include "../utils/utils.h"

#include <algorithm>

RandomBatchGenerator::RandomBatchGenerator(const DataSet& dataset, const Contaminator& contaminator,
                                           const Compressor& compressor)
    : dataset(dataset)
    , contaminator(contaminator)
    , compressor(compressor)
    , generator(1) {
}

std::string RandomBatchGenerator::get_random_string() {
    std::tuple<std::string, std::string, std::string> point_on_map = dataset.get_random_item(generator);
    static const std::string empty_token;
    const std::string& country = country_distribution(generator) == 0 ? std::get<0>(point_on_map) : empty_token;
    const std::string& city    = city_distribution(generator) == 0 ? std::get<1>(point_on_map) : empty_token;
    const std::string& street  = street_distribution(generator) == 0 ? std::get<2>(point_on_map) : empty_token;
    std::vector<std::string> components { country, city, street };
    std::shuffle(components.begin(), components.end(), generator);
    components.erase(std::remove(components.begin(), components.end(), ""), components.end());
    std::string message;
    for (const std::string& component : components) {
        if (!message.empty()) {
            message += "|";
        }
        message += component;
    }
    return message.empty() ? std::get<2>(point_on_map) : message;
}

void RandomBatchGenerator::generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size) {
    size_t message_size = compressor.get_message_size();
    std::fill(clean_batch, clean_batch + message_size * batch_size, Z_INT - A_INT + 1);
    std::fill(contaminated_batch, contaminated_batch + message_size * batch_size, Z_INT - A_INT + 1);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string clean_string = get_random_string();
        std::string compressed_clean_string = compressor.compress(clean_string);
        std::replace(clean_string.begin(), clean_string.end(), '|', ' ');
        std::string contaminated_string = contaminator.contaminate(clean_string);
        if (contaminated_string.length() > message_size) {
            contaminated_string = contaminated_string.substr(0, message_size);
        }
        int32_t shift = static_cast<int32_t>(i * message_size);
        std::transform(compressed_clean_string.begin(), compressed_clean_string.end(), clean_batch + shift, to_int);
        std::transform(contaminated_string.begin(), contaminated_string.end(), contaminated_batch + shift, to_int);
    }
}
