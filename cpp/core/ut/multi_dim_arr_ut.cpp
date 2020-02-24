#include <gtest/gtest.h>

#include <core/multi_dim_array.h>

#include <iostream>

// simply prints for now
TEST(MultiDimArray, Base) {
    using T = int;

    std::vector<int> sizes = {3, 4, 5};

    T* data = new T[3 * 4 * 5];
    multi_dim_array_idxs idxs({3, 4, 5});

    MultiDimArray<3, int> arr(data, &idxs, 0);

    int x = 0;

    for (int i = 0; i < sizes[0]; ++i) {
        for (int j = 0; j < sizes[1]; ++j) {
            for (int k = 0; k < sizes[2]; ++k) {
                data[i * 4 * 5 + j * 5 + k] = x++;
            }
        }
    }

    for (int i = 0; i < sizes[0]; ++i) {
        std::cout << i << std::endl;
        for (int j = 0; j < sizes[1]; ++j) {
            for (int k = 0; k < sizes[2]; ++k) {
                std::cout << std::setw(3) << arr[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}