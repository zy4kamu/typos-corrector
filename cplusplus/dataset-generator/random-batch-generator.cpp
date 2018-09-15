#include "random-batch-generator.h"

#include "../utils/utils.h"

#include <algorithm>

RandomBatchGenerator::RandomBatchGenerator(const DataSet& dataset, const Contaminator& contaminator, size_t message_size)
    : dataset(dataset)
    , contaminator(contaminator)
    , message_size(message_size)
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
            case 11:  // city
                if (city_distribution(generator) == 0) {
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
            default: // 12, 13, 15
                if (unknown_distribution(generator) == 0) {
                    components.push_back(entity->name);
                }
                break;
            }
        }

        // return tokens in random order
        std::shuffle(components.begin(), components.end(), generator);

        // allow to discard small parts of tokens: the netherlands -> netherlands
        for (std::string& component : components) {
            drop_articles(component);
        }

        // take prefix
        size_t prefix_length_to_achieve = std::max(static_cast<size_t>(5),
                                                   std::min(prefix_distribution(generator), message_size));
        size_t current_prefix_length = 0;
        std::vector<std::string> prefix_components;
        for (const std::string& component : components) {
            if (current_prefix_length + component.length() >= prefix_length_to_achieve) {
                prefix_components.push_back(component.substr(0, prefix_length_to_achieve - current_prefix_length));
                break;
            } else {
                prefix_components.emplace_back(component);
            }
            current_prefix_length += component.length();
        }

        if (!prefix_components.empty()) {
            return prefix_components;
        }
    }
}

void RandomBatchGenerator::drop_articles(std::string& token) {
    std::vector<std::string> splitted = split(token, ' ');
    if (splitted.size() < 2) {
        return;
    }
    if (std::find_if(splitted.begin(), splitted.end(), [](const std::string& part) { return part.length() < 4; }) == splitted.end())
    {
        return;
    }
    std::string dropped;
    for (const std::string& part : splitted) {
        if (part.length() > 3 || article_distribution(generator) == 0) {
            if (!dropped.empty()) {
                dropped += " ";
            }
            dropped += part;
        }
    }
    if (!dropped.empty()) {
        dropped.swap(token);
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
            message += component;
        }
    }
    if (message.length() > message_size) {
        message = message.substr(0, message_size);
    }
    return message;
}

void RandomBatchGenerator::generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size) {
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
