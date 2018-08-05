#include "sum-counter.h"

#include <iostream>

int main() {
    SumCounter counter(1024);
    std::vector<cl_float> data(1024, 1);
    std::cout << counter.calculate(data) << std::endl;
    std::cout << counter.exp(data)[0] << std::endl;
}