#include <funcs/soft_tree_function_cuda.h>
#include <gtest/gtest.h>

TEST(A, SoftTreeCuda) {
    EXPECT_EQ(2 * 2, 4);
    SoftTreeCuda softTreeCuda;
    softTreeCuda.testy_test();
}