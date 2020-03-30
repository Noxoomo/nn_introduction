#include <gtest/gtest.h>

#include <core/matrix.h>
#include <core/vec_factory.h>

TEST(MatrixTest, Transpose) {
    Vec tmp = VecFactory::fromVector({1, 2, 3, 4, 5, 6});
    Mx X(tmp, 2, 3);
    Mx XT = X.T();

    std::cout << "X: " << X << std::endl;
    std::cout << "XT: " << XT << std::endl;

    EXPECT_EQ(X.ydim(), XT.xdim());
    EXPECT_EQ(X.xdim(), XT.ydim());

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(X.get(j, i), XT.get(i, j));
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

TEST(MatrixTest, MultScalar) {
    Vec tmpA = VecFactory::fromVector({1, 2, 3, 4});
    Mx A(tmpA, 2, 2);

    Mx B = A * 2.5;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_EQ(A.get(i, j) * 2.5, B.get(i, j));
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

TEST(MatrixTest, AddColumn) {
    Vec tmp = VecFactory::fromVector({1, 2, 3, 4, 5, 6});
    Mx A(tmp, 3, 2);

    Vec col = VecFactory::fromVector({0.1, 0.2, 0.3});
    A.addColumn(col);

    EXPECT_EQ(3, A.ydim());
    EXPECT_EQ(3, A.xdim());

    int cnt = 0;

    for (int i = 0; i < A.ydim(); ++i) {
        for (int j = 0; j < A.xdim(); ++j) {
            if (j != 0) {
                ++cnt;
                EXPECT_DOUBLE_EQ(cnt, A.get(j, i));
            } else {
                EXPECT_DOUBLE_EQ((i + 1) * 0.1, A.get(j, i));
            }
        }
    }
}
