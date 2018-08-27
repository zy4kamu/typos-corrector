#include "dataset.h"

#include <fstream>

DataSet::DataSet(const boost::filesystem::path& input_folder) {
    std::string street;
    boost::filesystem::directory_iterator end_iter;

    // iterate over countries
    for (boost::filesystem::directory_iterator country_iter(input_folder); country_iter != end_iter; ++country_iter) {
        boost::filesystem::path country_path = country_iter->path();

        // iterate over cities
        keys.push_back(country_path.filename().string());
        values.emplace_back();
        CitiesStreets& cities = values.back();
        for (boost::filesystem::directory_iterator city_iter(country_path); city_iter != end_iter; ++city_iter) {
            boost::filesystem::path city_path = city_iter->path();
            cities.keys.push_back(city_path.filename().string());

            // iterate over streets
            cities.values.emplace_back();
            Streets& streets = cities.values.back();
            std::ifstream reader(city_path.string());
            while (getline(reader, street)) {
                streets.values.push_back(std::move(street));
            }
            streets.distribution = std::uniform_int_distribution<size_t>(0, streets.size() - 1);
            cities.number_of_streets += streets.size();
        }

        // distribution over cities
        std::vector<size_t> counters;
        for (const Streets& item : cities.values) {
            counters.push_back(item.size());
        }
        cities.distribution = std::discrete_distribution<size_t>(counters.begin(), counters.end());
    }

    // distribution over countries
    std::vector<size_t> counters;
    for (const CitiesStreets& item : values) {
        counters.push_back(item.size());
    }
    distribution = std::discrete_distribution<size_t>(counters.begin(), counters.end());
}

std::tuple<std::string, std::string, std::string> DataSet::get_random_item(std::mt19937& generator) {
    size_t index = distribution(generator);
    const std::string& country = keys[index];
    CitiesStreets& cities_streets = values[index];

    index = cities_streets.distribution(generator);
    const std::string& city = cities_streets.keys[index];
    Streets& streets = cities_streets.values[index];

    index = streets.distribution(generator);
    const std::string& street = streets.values[index];

    return std::make_tuple(country, city, street);
}
