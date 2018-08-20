#pragma once

#include "common.h"

#include <memory>
#include <map>
#include <vector>

#include <boost/filesystem/path.hpp>

namespace NNetworkHypoSearcher {

enum class HypoNodeType {
    Default,
    AllTransitions
};

struct HypoNode {
};

class HypoKeeper {
public:
    HypoKeeper(const boost::filesystem::path& first_mistake_file);
private:
    std::vector<cl_float> first_mistake_statistics;
};

} // namespace NNetworkHypoSearcher
