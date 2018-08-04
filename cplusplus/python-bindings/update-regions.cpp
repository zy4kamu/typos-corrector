#include "update-regions.h"

#include <algorithm>
#include <cassert>
#include <fstream>

UpdateRegionSet::UpdateRegionSet(const boost::filesystem::path& input_folder) {
    boost::filesystem::directory_iterator end_iter;
    for (boost::filesystem::directory_iterator iter(input_folder); iter != end_iter; ++iter) {
        boost::filesystem::path path = iter->path();
        assert(boost::filesystem::is_regular_file(path));
        update_regions.emplace_back();
        UpdateRegion& current_update_region = update_regions.back();
        current_update_region.name = path.filename().string();
        std::ifstream reader(path.string());
        std::string line;
        while (getline(reader, line)) {
            current_update_region.tokens.push_back(std::move(line));
        }
        assert(!current_update_region.tokens.empty());
    }
}
