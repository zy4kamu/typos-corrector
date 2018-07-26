#include "python-bindings.h"

#include <cassert>
#include <memory>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/make_unique.hpp>

#include "compressor.h"
#include "contaminator.h"
#include "random-batch-generator.h"
#include "update-regions.h"

extern "C" {

boost::filesystem::path               UPDATE_REGIONS_FOLDER;
std::unique_ptr<Compressor>           COMPRESSOR;
std::unique_ptr<Contaminator>         CONTAMINATOR;
std::unique_ptr<UpdateRegionSet>      UPDATE_REGION_SET;
std::unique_ptr<RandomBatchGenerator> BATCH_GENERATOR;

void set_update_regions_folder(const char* input_folder) {
    assert(UPDATE_REGIONS_FOLDER.empty());
    UPDATE_REGIONS_FOLDER = input_folder;
    assert(boost::filesystem::exists(UPDATE_REGIONS_FOLDER));
}

void create_update_regions_set() {
    assert(!UPDATE_REGIONS_FOLDER.empty());
    assert(!UPDATE_REGION_SET);
    UPDATE_REGION_SET = boost::make_unique<UpdateRegionSet>(UPDATE_REGIONS_FOLDER);
}

void create_contaminator(double mistake_probability) {
    assert(!CONTAMINATOR);
    CONTAMINATOR = boost::make_unique<Contaminator>(mistake_probability);
}

void create_compressor() {
    assert(UPDATE_REGION_SET);
    assert(!COMPRESSOR);
    COMPRESSOR = boost::make_unique<Compressor>(*UPDATE_REGION_SET);
}

void decompress(const char* token, char* output) {
    assert(COMPRESSOR);
    const std::vector<std::string>& decompressed = COMPRESSOR->decompress(token);
    std::stringstream stream;
    for (const std::string& decompresed_token : decompressed) {
        stream << decompresed_token << "|";
    }
    const std::string concatenated = stream.str();
    if (!concatenated.empty()) {
        std::memcpy(output, concatenated.c_str(), concatenated.length() - 1);
    }
}

void find_by_prefix(const char* prefix, size_t max_size, char* output) {
    assert(COMPRESSOR);
    std::vector<std::string> decompressed = COMPRESSOR->find_by_prefix(prefix, max_size);
    std::stringstream stream;
    for (const std::string& decompresed_token : decompressed) {
        stream << decompresed_token << "|";
    }
    const std::string concatenated = stream.str();
    if (!concatenated.empty()) {
        std::memcpy(output, concatenated.c_str(), concatenated.length() - 1);
    }
}

void create_random_batch_generator() {
    assert(UPDATE_REGION_SET);
    assert(!BATCH_GENERATOR);
    BATCH_GENERATOR = boost::make_unique<RandomBatchGenerator>(*UPDATE_REGION_SET, *CONTAMINATOR, *COMPRESSOR);
}

int32_t generate_random_batch_on_one_update_region(int32_t* clean_batch, int32_t* contaminated_batch,
                                                   size_t message_size, size_t batch_size) {
    assert(BATCH_GENERATOR);
    return BATCH_GENERATOR->generate_random_batch_on_one_ur(clean_batch, contaminated_batch, message_size, batch_size);
}

void generate_random_batch_on_all_update_regions(int32_t* clean_batch, int32_t* contaminated_batch,
                                                    size_t message_size, size_t batch_size) {
    assert(BATCH_GENERATOR);
    BATCH_GENERATOR->generate_random_batch_on_all_urs(clean_batch, contaminated_batch, message_size, batch_size);
}

} // extern "C"
