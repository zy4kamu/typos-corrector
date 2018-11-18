#pragma once

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "i-database-requester.h"
#include "../dataset/dataset.h"
#include "../prefix-tree/prefix-tree.h"

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
    HypoNode(const HypoNode* parent, float_type logit, std::string&& prefix)
        : parent(parent), logit(logit), prefix(std::move(prefix)) {
    }

    const HypoNode* parent;
    float_type logit;
    std::string prefix;
    std::vector<HypoNode> transitions;
    PrefixTreeState prefix_tree_state;
    CompressedLSTMCellCPU::InternalState network_state;
};

struct HypoNodePointerComparator {
    bool operator()(const HypoNode* first, const HypoNode* second) const;
};

using AutomataNodesSet = std::set<HypoNode*, HypoNodePointerComparator>;

class HypoSearcher {
public:
    HypoSearcher(const std::string& lstm_folder);
    std::vector<std::string> cover_probability(const std::string& input, float_type target_probability,
                                               size_t max_attempts, PrefixTreeMaster& prefix_tree);
    void load();
    void unload();
    bool is_loaded() const;
    void initialize(const std::string& input);
    const std::string& generate_next_hypo();
    bool check_hypo_in_database(IDataBaseRequester& requester, std::string& levenstein_correction);
    float_type get_probability_not_to_correct() const;
private:
    void read_first_mistake_statistics(const std::string& first_mistake_file);

    std::string             lstm_folder;
    NetworkAutomata         automata;
    std::vector<float_type> first_mistake_statistics;
    size_t                  max_prefix_length;
    std::string             current_hypo;
    std::vector<float_type> current_probabilities;
    AutomataNodesSet        nodes_to_process;
    HypoNode                root;
    float_type              probability_not_to_correct;
    std::string             initial_input;
    size_t                  current_levenstein;
};

} // namespace NNetworkHypoSearcher
