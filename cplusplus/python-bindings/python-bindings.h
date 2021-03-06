#pragma once

#include <cstdint>
#include <cstring>

extern "C" {

void set_dataset_folder(const char* input_folder);
void reset();
void create_dataset(size_t split_index);
void create_contaminator(const char* ngrams_file, double mistake_probability);
void find_by_prefix(const char* prefix, size_t max_size, char* output);
void create_random_batch_generator(size_t message_size);
void generate_random_batch(int32_t* clean_batch, int32_t* contaminated_batch, size_t batch_size);
size_t levenstein(const char* first, const char* second, size_t message_size);
void create_prefix_tree_builder();
void add_to_prefix_tree_builder(const char* message);
void finalize_prefix_tree_builder(const char* output_file);

} // extern "C"
