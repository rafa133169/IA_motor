#include <gtest/gtest.h>
#include "nevrocpp/core/Tensor.hpp"
#include "nevrocpp/layers/ReLU.hpp"

using namespace nevrocpp;

TEST(ReLUTest, EliminatesNegatives) {
    core::Tensor t(1, 5);

    t.set(0, 0, -1.0f);
    t.set(0, 1, 0.5f);
    t.set(0, 2, -3.0f);
    t.set(0, 3, 2.0f);
    t.set(0, 4, 0.0f);

    layers::ReLU relu;
    relu.apply_inplace(t);

    EXPECT_FLOAT_EQ(t.get(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t.get(0, 1), 0.5f);
    EXPECT_FLOAT_EQ(t.get(0, 2), 0.0f);
    EXPECT_FLOAT_EQ(t.get(0, 3), 2.0f);
    EXPECT_FLOAT_EQ(t.get(0, 4), 0.0f);
}
