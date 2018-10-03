#pragma once

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "i-database-requester.h"
#include "../dataset/dataset.h"

#ifdef USE_OPENCL
    #include "network-automata-gpu.h"
    using NetworkAutomata = NNetworkHypoSearcher::NetworkAutomataGPU;
#else
    #include "network-automata-cpu.h"
    using NetworkAutomata = NNetworkHypoSearcher::NetworkAutomataCPU;
#endif

namespace NNetworkHypoSearcher {

struct HypoNode {
    HypoNode() = default;
    HypoNode(char letter, float_type logit, std::string&& prefix)
        : letter(letter), logit(logit), prefix(std::move(prefix)) {
    }

    char letter;
    float_type logit;
    std::string prefix; // TODO: replace with NN state
    std::vector<HypoNode> transitions;
};

struct HypoNodePointerComparator {
    bool operator()(const HypoNode* first, const HypoNode* second) const;
};

using AutomataNodesSet = std::set<HypoNode*, HypoNodePointerComparator>;

class HypoSearcher {
public:
    HypoSearcher(const std::string& lstm_folder);
    void initialize(const std::string& input);
    const std::string& generate_next_hypo();
    bool check_hypo_in_database(IDataBaseRequester& requester);
    float_type get_probability_not_to_correct() const;
private:
    void read_first_mistake_statistics(const std::string& first_mistake_file);
    void reset();

    NetworkAutomata         automata;
    std::vector<float_type> first_mistake_statistics;
    size_t                  max_prefix_length;
    std::string             current_hypo;
    std::vector<float_type> current_probabilities;
    AutomataNodesSet        nodes_to_process;
    HypoNode                root;
    float_type              probability_not_to_correct;
};

} // namespace NNetworkHypoSearcher
