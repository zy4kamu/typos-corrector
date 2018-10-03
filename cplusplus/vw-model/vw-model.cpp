#include "vw-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

namespace NVWModel {

namespace {

const size_t COUNTRY_SET_HASH_SIZE = 4096 * 4;

std::vector<float_type> read_file(const std::string& filename) {
    std::vector<float_type> data;
    std::ifstream file(filename, std::ios::binary);
    float_type item;
    while (file.read(reinterpret_cast<char*>(&item), sizeof(float_type))) {
        data.push_back(item);
    }
    return data;
}

std::vector<size_t> create_features(const std::string& message) {
    std::vector<size_t> features;
    for (size_t length = 1; length <= message.size(); ++length) {
        for (size_t i = 0; i + length < message.length(); ++i) {
            const std::string feature = message.substr(i, length);
            features.push_back(std::hash<std::string>{}(feature) % COUNTRY_SET_HASH_SIZE);
        }
    }
    return features;
}

} // unonymous namespace

VWModel::VWModel(const std::string& input_folder) {
    std::ifstream labels_reader(input_folder + "/labels");
    std::string line;
    while (getline(labels_reader, line)) {
        labels.push_back(std::move(line));
    }
    weights = read_file(input_folder + "/data");
    num_labels = labels.size();
    num_features = weights.size() / num_labels - 1;
}

VWModel::MapType VWModel::predict(const std::string& message) const {
    // calculate logits
    std::vector<float_type> predictions(num_labels);
    std::vector<size_t> features = create_features(message);
    std::memcpy(predictions.data(), weights.data() + num_features * num_labels, sizeof(float_type) * num_labels);
    for (size_t feature : features) {
        const float_type* pointer = weights.data() + num_labels * feature;
        for (size_t i = 0; i < num_labels; ++i) {
            predictions[i] += pointer[i];
        }
    }

    // caclulate probabilities
    float_type max = std::numeric_limits<float_type>::min();
    for (float_type value : predictions) {
        max = std::max(max, value);
    }
    float_type sum = 0;
    for (float_type& value : predictions) {
        value -= max;
        value = std::exp(value);
        sum += value;
    }
    for (float_type& value : predictions) {
        value /= sum;
    }

    MapType result;
    for (size_t i = 0; i < num_labels; ++i) {
        result.insert(std::make_pair(predictions[i], labels[i]));
    }
    return result;
}

} // namespace NVWModel
