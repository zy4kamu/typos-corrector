#include "random-batch-generator.h"

const size_t MESSAGE_SIZE = 25;

int main() {
    std::mt19937 generator(1);
    DataSet dataset("/home/stepan/git-repos/typos-corrector/python/model/dataset");
    Contaminator contaminator("/home/stepan/git-repos/typos-corrector/python/model/ngrams", 0.2);
    Compressor compressor(dataset, MESSAGE_SIZE);
    RandomBatchGenerator batch_generator(dataset, contaminator, compressor);
    std::vector<int32_t> clean_tokens(MESSAGE_SIZE * 1000);
    std::vector<int32_t> contaminated_tokens(MESSAGE_SIZE * 1000);
    batch_generator.generate_random_batch(clean_tokens.data(), contaminated_tokens.data(), 1000);
}
