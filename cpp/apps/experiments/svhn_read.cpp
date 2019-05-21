#include "common.h"

#include <iostream>

int main() {
    using namespace experiments;

    char* args[] = {(char*)"svhn"};
    auto ds = readDataset(1, args);

    std::cout << ds.first.size() .value() << std::endl;
}