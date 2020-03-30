#include <gtest/gtest.h>

#include <core/multi_dim_array.h>

#include <iostream>

struct TestStruct {
    TestStruct() {
        x.resize(size, 0.);
        for (int i = 0; i < size; ++i) {
            x[i] = i;
        }
    }

    TestStruct(const TestStruct& other) = default;

    TestStruct(TestStruct&& other) {
        std::cout << "move" << std::endl;
    }

    TestStruct(const TestStruct&& other) {
        std::cout << "move" << std::endl;
    }

    int size = 10;
    std::vector<float> x;
};


// simply prints for now
TEST(MultiDimArray, Base) {
    using T = int;

    std::vector<int> sizes = {3, 4, 5};
    MultiDimArray<3, int> arr(sizes);

    int x = 0;

    for (int i = 0; i < sizes[0]; ++i) {
        for (int j = 0; j < sizes[1]; ++j) {
            for (int k = 0; k < sizes[2]; ++k) {
                arr[i][j][k] = x++;
            }
        }
    }

    MultiDimArray<2, int> arr1 = arr[1].copy();

    for (int j = 0; j < sizes[1]; ++j) {
        for (int k = 0; k < sizes[2]; ++k) {
            std::cout << std::setw(3) << arr1[j][k] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << 0 << std::endl;
    TestStruct defaultVal;
    MultiDimArray<5, TestStruct> arrCheck({51, 13, 3, 22, 5}, defaultVal);
    std::cout << 1 << std::endl;
    auto arr2 = arrCheck[2][2][2][2].copy();
    std::cout << 2 << std::endl;
    auto arr3 = std::move(arr2);
    std::cout << 3 << std::endl;
    std::cout << "size=" << arr3[2].size << std::endl;
    for (int k = 0; k < arr3[2].size; ++k) {
        std::cout << std::setw(3) << arr3[2].x[k] << " ";
    }
}