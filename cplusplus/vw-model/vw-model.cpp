#include "vw-model.h"

#include <cstring>
#include <fstream>
#include <iterator>

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
    predictions.resize(labels.size(), 0);
    weights = read_file(input_folder + "/data");
    num_labels = labels.size();
    num_features = weights.size() / num_labels - 1;
}

const std::vector<float_type>& VWModel::predict(const std::string& message) const {
    std::vector<size_t> features = create_features(message);
    std::memcpy(predictions.data(), weights.data() + num_features * num_labels, sizeof(float_type) * num_labels);
    for (size_t feature : features) {
        const float_type* pointer = weights.data() + num_labels * feature;
        for (size_t i = 0; i < num_labels; ++i) {
            predictions[i] += pointer[i];
        }
    }
    return predictions;
}

} // namespace NVWModel
