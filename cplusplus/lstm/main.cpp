#include "lstm.h"

#include <iostream>

int main() {
    LSTMCell counter(27, 1024);
    std::vector<cl_float> data(1024, 1);
    std::cout << counter.exp(data)[0] << std::endl;
}
