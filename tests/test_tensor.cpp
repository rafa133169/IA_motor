// Archivo vacío
#include <gtest/gtest.h>
#include "nevrocpp/core/Tensor.hpp"
#include "nevrocpp/losses/MSE.hpp"

using namespace nevrocpp;

TEST(TensorTest, SetAndGet) {
	core::Tensor t(2, 2);
	t.set(0, 0, 1.0f);
	t.set(0, 1, 2.0f);
	t.set(1, 0, 3.0f);
	t.set(1, 1, 4.0f);
	EXPECT_FLOAT_EQ(t.get(0, 0), 1.0f);
	EXPECT_FLOAT_EQ(t.get(0, 1), 2.0f);
	EXPECT_FLOAT_EQ(t.get(1, 0), 3.0f);
	EXPECT_FLOAT_EQ(t.get(1, 1), 4.0f);
}

TEST(TensorTest, DotProduct) {
	core::Tensor a(2, 3);
	core::Tensor b(3, 2);
	// a = [1 2 3; 4 5 6]
	a.set(0, 0, 1.0f); a.set(0, 1, 2.0f); a.set(0, 2, 3.0f);
	a.set(1, 0, 4.0f); a.set(1, 1, 5.0f); a.set(1, 2, 6.0f);
	// b = [7 8; 9 10; 11 12]
	b.set(0, 0, 7.0f); b.set(0, 1, 8.0f);
	b.set(1, 0, 9.0f); b.set(1, 1, 10.0f);
	b.set(2, 0, 11.0f); b.set(2, 1, 12.0f);
	core::Tensor c = a.dot(b);
	// c = [58 64; 139 154]
	EXPECT_FLOAT_EQ(c.get(0, 0), 58.0f);
	EXPECT_FLOAT_EQ(c.get(0, 1), 64.0f);
	EXPECT_FLOAT_EQ(c.get(1, 0), 139.0f);
	EXPECT_FLOAT_EQ(c.get(1, 1), 154.0f);
}


TEST(MSETest, ComputeMSE) {
	core::Tensor y_true(1, 3);
	core::Tensor y_pred(1, 3);
	y_true.set(0, 0, 1.0f); y_true.set(0, 1, 2.0f); y_true.set(0, 2, 3.0f);
	y_pred.set(0, 0, 2.0f); y_pred.set(0, 1, 2.0f); y_pred.set(0, 2, 4.0f);
	// MSE = ((1-2)^2 + (2-2)^2 + (3-4)^2) / 3 = (1 + 0 + 1) / 3 = 0.666...
	float mse = losses::MSE::compute(y_true, y_pred);
	EXPECT_NEAR(mse, 0.6667f, 1e-4f);
}

TEST(MSETest, Gradient) {
	core::Tensor y_true(1, 3);
	core::Tensor y_pred(1, 3);
	y_true.set(0, 0, 1.0f); y_true.set(0, 1, 2.0f); y_true.set(0, 2, 3.0f);
	y_pred.set(0, 0, 2.0f); y_pred.set(0, 1, 2.0f); y_pred.set(0, 2, 4.0f);
	// grad = 2 * (y_pred - y_true) = [2, 0, 2]
	core::Tensor grad = losses::MSE::gradient(y_true, y_pred);
	EXPECT_FLOAT_EQ(grad.get(0, 0), 2.0f);
	EXPECT_FLOAT_EQ(grad.get(0, 1), 0.0f);
	EXPECT_FLOAT_EQ(grad.get(0, 2), 2.0f);
}
