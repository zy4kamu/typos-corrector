#include "lstm.h"

#include <iostream>

int main() {
    LSTMCell counter("/home/stepan/git-repos/typos-corrector/python/model/parameters", 27, 128, 512);
    std::vector<cl_float> data(512, 1);
    std::cout << counter.exp(data)[0] << std::endl;
}
