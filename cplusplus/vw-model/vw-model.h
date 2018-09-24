#include <string>
#include <vector>

namespace NVWModel {

using float_type = float;

class VWModel {
public:
    VWModel(const std::string& input_folder);
    const std::vector<float_type>& predict(const std::string& message) const;
    const std::string& label(size_t index) const { return labels[index]; }
private:
    std::vector<float_type> weights;
    std::vector<std::string> labels;
    mutable std::vector<float_type> predictions;

    size_t num_features;
    size_t num_labels;
};

} // namespace NVWModel
