#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace NVWModel {

using float_type = float;

class VWModel {
public:
    using MapType = std::multimap<float_type, size_t, std::greater<float_type>>;
    VWModel(const std::string& input_folder);
    MapType predict(const std::string& message, const std::unordered_map<std::string, size_t>& label_to_index) const;
    const std::string& label(size_t index) const { return labels[index]; }
private:
    std::vector<float_type> weights;
    std::vector<std::string> labels;

    size_t num_features;
    size_t num_labels;
};

} // namespace NVWModel
