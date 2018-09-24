#include "random-batch-generator.h"

const size_t MESSAGE_SIZE = 25;

int main() {
    std::mt19937 generator(1);
    DataSet dataset("/home/stepan/git-repos/typos-corrector/python/model/dataset", 0);
    Contaminator contaminator("/home/stepan/git-repos/typos-corrector/python/model/ngrams", 0.2);
    RandomBatchGenerator batch_generator(dataset, contaminator, MESSAGE_SIZE);
    batch_generator.generate_country_dataset("/home/stepan/country-dataset/dataset/", 100000000, 100000000);
}
