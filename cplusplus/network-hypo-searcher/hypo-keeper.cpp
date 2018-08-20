#include "hypo-keeper.h"

#include <fstream>

namespace NNetworkHypoSearcher {

HypoKeeper::HypoKeeper(const boost::filesystem::path& first_mistake_file) {
    std::ifstream reader(first_mistake_file.string());
    std::string line;
    while (getline(reader, line)) {
        first_mistake_statistics.push_back(static_cast<float_type>(std::stof(line) + 1.0));
    }
    for (size_t i = first_mistake_statistics.size() - 1; i + 1 != 0; --i) {
        first_mistake_statistics[i] += first_mistake_statistics[i + 1];
    }
    for (float_type& item : first_mistake_statistics) {
        item /= first_mistake_statistics.back();
        item = std::log(item);
    }
}

} // namespace NNetworkHypoSearcher
