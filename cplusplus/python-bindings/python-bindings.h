#pragma once

#include <cstdint>
#include <cstring>

extern "C" {

void set_dataset_folder(const char* input_folder);
void create_dataset();
void create_contaminator(const char* ngrams_file, double mistake_probability);
void create_compressor(size_t message_size);
void decompress(const char* token, char* output);
void find_by_prefix(const char* prefix, size_t max_size, char* output);
void create_random_batch_generator();
void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size);
size_t levenstein(const char* first, const char* second, size_t message_size);

} // extern "C"
