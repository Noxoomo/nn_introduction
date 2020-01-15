#include <gtest/gtest.h>

#include <core/matrix.h>
#include <core/vec_factory.h>

TEST(MatrixTest, Transpose) {
    Vec tmp = VecFactory::fromVector({1, 2, 3, 4});
    Mx X(tmp, 2, 2);
    Mx XT = X.T();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_EQ(X.get(i, j), XT.get(j, i));
        }
    }
}

TEST(MatrixTest, Multiplication) {
    Vec tmpA = VecFactory::fromVector({1, 2, 3, 4, 5, 6});
    Vec tmpB = VecFactory::fromVector({6, 5, 4, 3, 2, 1});

    Mx A(tmpA, 2, 3);
    Mx B(tmpB, 3, 2);

    Mx C = A * B;

    Vec tmpExpectedRes = VecFactory::fromVector({20, 14, 56, 41});
    Mx ExpectedRes(tmpExpectedRes, 2, 2);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_EQ(ExpectedRes.get(i, j), C.get(i, j));
        }
    }
}

TEST(MatrixTest, Inverse) {
    Vec tmp = VecFactory::fromVector({1, 4, 5, 3});

    Mx A(tmp, 2, 2);
    Mx B = A.inverse();

    Vec tmpExpectedRes = VecFactory::fromVector({-3.0/17, 4.0/17, 5.0/17, -1.0/17});
    Mx ExpectedRes(tmpExpectedRes, 2, 2);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(ExpectedRes.get(i, j), B.get(i, j));
        }
    }
}
