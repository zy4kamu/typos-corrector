#pragma once

#include <string>
#include <vector>

#include <boost/filesystem.hpp>

struct UpdateRegion {
    std::string name;
    std::vector<std::string> tokens;
};

struct UpdateRegionSet {
    UpdateRegionSet(const boost::filesystem::path& input_folder);
    std::vector<UpdateRegion> update_regions;
};
