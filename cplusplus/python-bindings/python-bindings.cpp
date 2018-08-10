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

void create_contaminator(const char* ngrams_file, double mistake_probability) {
    assert(!CONTAMINATOR);
    CONTAMINATOR = boost::make_unique<Contaminator>(ngrams_file, mistake_probability);
}

void create_compressor(size_t message_size) {
    assert(UPDATE_REGION_SET);
    assert(!COMPRESSOR);
    COMPRESSOR = boost::make_unique<Compressor>(*UPDATE_REGION_SET, message_size);
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

int32_t generate_random_batch_on_one_update_region(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size) {
    assert(BATCH_GENERATOR);
    return BATCH_GENERATOR->generate_random_batch_on_one_ur(clean_batch, contaminated_batch, batch_size);
}

void generate_random_batch_on_all_update_regions(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size) {
    assert(BATCH_GENERATOR);
    BATCH_GENERATOR->generate_random_batch_on_all_urs(clean_batch, contaminated_batch, batch_size);
}

size_t levenstein(const char* first, const char* second, size_t message_size) {
  if (message_size == 0) {
    return 0;
  }
  size_t grid_size = message_size + 1;
  std::vector<size_t> grid(grid_size * grid_size, 0);
  for (size_t i = 0; i < grid_size; ++i) {
    grid[grid_size * message_size + i] = grid[message_size + i * grid_size] = grid_size - i - 1;
  }
  for (size_t i = message_size - 1; i + 1 != 0; --i) {
    for (size_t j = message_size - 1; j + 1 != 0; --j) {
      if (first[i] == second[j]) {
        grid[i * grid_size + j] = grid[(i + 1) * grid_size + (j + 1)];
      } else {
        grid[i * grid_size + j] = 1 + std::min(grid[(i + 1) * grid_size + (j + 1)],
                                               std::min(grid[(i + 1) * grid_size + j], grid[i * grid_size + (j + 1)]));
      }
    }
  }
  return grid[0];
}

} // extern "C"
