#pragma once

#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>

class DataSet {
public:
    struct Entity {
        int32_t type;
        std::string name;
    };

    struct Node {
        int32_t name_index;
        int32_t count;
        mutable std::discrete_distribution<size_t> transition_distribution;
        std::vector<Node> transitions;
        std::unordered_map<int32_t, int32_t> tranisition_name_index_to_transition_index;
        std::vector<int32_t> house_numbers;

        Node* add(size_t index);
        void finalize();
    };

    DataSet(std::string country_folder, bool use_transitions = true);
    std::vector<const Entity*> get_random_item(std::mt19937& generator) const;
    const std::unordered_map<size_t, Entity>& content() const;
    std::vector<std::string> find_by_prefix(const std::string& prefix, size_t max_number) const;
private:
    std::unordered_map<size_t, Entity> index_to_name;
    mutable std::uniform_int_distribution<size_t> house_numbers_distribution;
    Node root;
    std::set<std::string> tokens;
};
