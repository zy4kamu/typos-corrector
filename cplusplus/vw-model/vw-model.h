#include <functional>
#include <map>
#include <string>
#include <vector>

namespace NVWModel {

using float_type = float;

class VWModel {
public:
    using MapType = std::multimap<float_type, std::string, std::greater<float_type>>;
    VWModel(const std::string& input_folder);
    MapType predict(const std::string& message) const;
    const std::string& label(size_t index) const { return labels[index]; }
private:
    std::vector<float_type> weights;
    std::vector<std::string> labels;

    size_t num_features;
    size_t num_labels;
};

} // namespace NVWModel
