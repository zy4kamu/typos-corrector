#pragma once

#include <cstdint>
#include <cstring>

extern "C" {

void set_update_regions_folder(const char* input_folder);
void create_update_regions_set();
void create_contaminator(const char* ngrams_file, double mistake_probability);
void create_compressor(size_t message_size);
void decompress(const char* token, char* output);
void find_by_prefix(const char* prefix, size_t max_size, char* output);
void create_random_batch_generator();
int32_t generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t message_size, size_t batch_size);

} // extern "C"
