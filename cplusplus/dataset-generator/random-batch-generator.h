#pragma once

#include <random>
#include <string>
#include <vector>

#include "contaminator.h"
#include "../dataset/dataset.h"

// TODO: capital letters, diacritics, so on
// TODO: bind it to navkit
// TODO: whole europe

class RandomBatchGenerator {
public:
    RandomBatchGenerator(const DataSet& dataset, const Contaminator& contaminator, size_t message_size);
    void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size);
    void generate_country_dataset(const std::string& output_folder, size_t train_size, size_t test_size);
private:
    void generate_country_dataset(const std::string& output_file, size_t size,
                                  std::map<std::string, size_t>& country_to_index);
    std::vector<std::string> generate_one_example(std::string* country = nullptr);
    std::string get_clean_string(const std::vector<std::string>& example, char separator='|') const;
    void drop_articles(std::string& token);
    std::vector<size_t> get_vw_features(const std::string& message) const;

    const DataSet&      dataset;
    const Contaminator& contaminator;
    size_t message_size;
    std::mt19937        generator;

    std::discrete_distribution<size_t>  country_distribution  { 0.05, 0.95 };
    std::discrete_distribution<size_t>  city_distribution     { 0.25, 0.75 };
    std::discrete_distribution<size_t>  district_distribution { 0.15, 0.85 };
    std::discrete_distribution<size_t>  index_distribution    { 0.05, 0.95 };
    std::discrete_distribution<size_t>  street_distribution   { 0.9, 0.1 };
    std::discrete_distribution<size_t>  house_distribution    { 0.3, 0.7 };
    std::discrete_distribution<size_t>  unknown_distribution  { 0.1, 0.9 };

    std::discrete_distribution<size_t>  article_distribution  { 0.75, 0.25 };
    std::geometric_distribution<size_t> prefix_distribution   { 1. / 15. };
};

