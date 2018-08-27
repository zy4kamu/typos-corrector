#pragma once

#include <boost/filesystem.hpp>

#include <map>
#include <string>
#include <utility>

class DataSet {
public:
    DataSet(const boost::filesystem::path& input_folder);
    std::tuple<std::string, std::string, std::string> get_random_item(std::mt19937& generator);
private:
    struct Streets {
        std::vector<std::string> values;
        std::uniform_int_distribution<size_t> distribution;
        size_t size() const { return values.size(); }
    };

    struct CitiesStreets {
        std::vector<std::string> keys; // cities
        std::vector<Streets> values;   // streets
        std::discrete_distribution<size_t> distribution;
        size_t size() const { return number_of_streets; }
    private:
        size_t number_of_streets;
        friend class DataSet;
    };

    std::vector<std::string> keys;     // countries
    std::vector<CitiesStreets> values; // cities
    std::discrete_distribution<size_t> distribution;
};
