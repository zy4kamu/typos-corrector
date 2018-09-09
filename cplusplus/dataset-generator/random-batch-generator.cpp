#include "random-batch-generator.h"

#include "../dataset/compressor.h"
#include "../utils/utils.h"

#include <algorithm>

RandomBatchGenerator::RandomBatchGenerator(const DataSet& dataset, const Contaminator& contaminator,
                                           const Compressor& compressor)
    : dataset(dataset)
    , contaminator(contaminator)
    , compressor(compressor)
    , generator(1) {
}

std::vector<std::string> RandomBatchGenerator::generate_one_example() {
    while (true) {
        std::vector<const DataSet::Entity*> entities = dataset.get_random_item(generator);
        std::vector<std::string> components;
        bool added_index = false; // there are extended and normal indexes; only one of them can be requested in one message
        for (const DataSet::Entity* entity : entities) {
            switch (entity->type) {
            case 3:   // country
                if (country_distribution(generator) == 0) {
                    components.push_back(entity->name);
                }
                break;
            case 23:  // index
            case 30:  // extended index
                if (!added_index) {
                    if (index_distribution(generator) == 0) {
                        added_index = true;
                        components.push_back(entity->name);
                    }
                }
                break;
            case 100: // street
                if (street_distribution(generator) == 0) {
                    components.push_back(entity->name);
                }
                break;
            case 101: // house number
                if (house_distribution(generator) == 0) {
                    components.push_back(entity->name);
                }
                break;
            }
        }
        if (!components.empty()) {
            return components;
        }
    }
}

std::string RandomBatchGenerator::get_clean_string(const std::vector<std::string>& example) const {
    std::string message;
    for (const std::string& component : example) {
        if (!message.empty()) {
            message += "|";
        }
        if (contains_digit(component)) {
            message += component;
        } else {
            message += compressor.compress(component);
        }
    }
    return message;
}

void RandomBatchGenerator::generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size) {
    size_t message_size = compressor.get_message_size();
    std::fill(clean_batch, clean_batch + message_size * batch_size, to_int(' '));
    std::fill(contaminated_batch, contaminated_batch + message_size * batch_size, to_int(' '));
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<std::string> example = generate_one_example();
        std::string compressed_clean_string = get_clean_string(example);
        std::string contaminated_string = contaminator.contaminate(example, message_size);
        int32_t shift = static_cast<int32_t>(i * message_size);
        std::transform(compressed_clean_string.begin(), compressed_clean_string.end(), clean_batch + shift, to_int);
        std::transform(contaminated_string.begin(), contaminated_string.end(), contaminated_batch + shift, to_int);
    }
}
