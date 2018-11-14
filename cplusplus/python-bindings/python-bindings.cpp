#include "python-bindings.h"

#include <cassert>
#include <fstream>
#include <memory>
#include <sstream>

#include <boost/make_unique.hpp>

#include "../dataset/dataset.h"
#include "../dataset-generator/contaminator.h"
#include "../dataset-generator/random-batch-generator.h"
#include "../network-hypo-searcher/utils.h"
#include "../prefix-tree/prefix-tree-builder.h"

extern "C" {

std::string                           DATASET_FOLDER;
std::unique_ptr<Contaminator>         CONTAMINATOR;
std::unique_ptr<DataSet>              DATASET;
std::unique_ptr<RandomBatchGenerator> BATCH_GENERATOR;
std::unique_ptr<PrefixTreeBuilder>    PREFIX_TREE_BUILDER;

void set_dataset_folder(const char* input_folder) {
    assert(DATASET_FOLDER.empty());
    DATASET_FOLDER = input_folder;
}

void reset() {
    BATCH_GENERATOR.reset();
    CONTAMINATOR.reset();
    DATASET.reset();
}

void create_dataset(size_t split_index) {
    assert(!DATASET_FOLDER.empty());
    assert(!DATASET);
    DATASET = boost::make_unique<DataSet>(DATASET_FOLDER, split_index);
}

void create_contaminator(const char* ngrams_file, double mistake_probability) {
    assert(!CONTAMINATOR);
    CONTAMINATOR = boost::make_unique<Contaminator>(ngrams_file, mistake_probability);
}

void find_by_prefix(const char* prefix, size_t max_size, char* output) {
    assert(DATASET);
    std::vector<std::string> decompressed = DATASET->find_by_prefix(prefix, max_size);
    std::stringstream stream;
    for (const std::string& decompresed_token : decompressed) {
        stream << decompresed_token << "|";
    }
    const std::string concatenated = stream.str();
    if (!concatenated.empty()) {
        std::memcpy(output, concatenated.c_str(), concatenated.length() - 1);
    }
}

void create_random_batch_generator(size_t message_size) {
    assert(DATASET);
    assert(!BATCH_GENERATOR);
    BATCH_GENERATOR = boost::make_unique<RandomBatchGenerator>(*DATASET, *CONTAMINATOR, message_size);
}

void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size) {
    assert(BATCH_GENERATOR);
    BATCH_GENERATOR->generate_random_batch(clean_batch, contaminated_batch, batch_size);
}

size_t levenstein(const char* first, const char* second, size_t message_size) {
    return levenstein_distance(first, second, message_size);
}

void create_prefix_tree_builder() {
    PREFIX_TREE_BUILDER = boost::make_unique<PrefixTreeBuilder>();
}

void add_to_prefix_tree_builder(const char* message) {
    PREFIX_TREE_BUILDER->add(message);
}

void finalize_prefix_tree_builder(const char* output_file) {
    std::vector<char> content = PREFIX_TREE_BUILDER->to_string();
    std::ofstream writer(output_file, std::ios::binary);
    writer.write(content.data(), content.size());
}

} // extern "C"
