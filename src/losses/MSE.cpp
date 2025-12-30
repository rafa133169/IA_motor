#include "nevrocpp/losses/MSE.hpp"
#include <stdexcept>

namespace nevrocpp::losses {

float MSE::compute(const core::Tensor& y_true, const core::Tensor& y_pred) {
	int rows = y_true.getRows();
	int cols = y_true.getCols();
	if (rows != y_pred.getRows() || cols != y_pred.getCols()) {
		throw std::invalid_argument("Dimensiones incompatibles para MSE");
	}
	float sum = 0.0f;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			float diff = y_true.get(i, j) - y_pred.get(i, j);
			sum += diff * diff;
		}
	}
	return sum / (rows * cols);
}

} // namespace nevrocpp::losses

core::Tensor MSE::gradient(const core::Tensor& y_true, const core::Tensor& y_pred) {
	int rows = y_true.getRows();
	int cols = y_true.getCols();
	if (rows != y_pred.getRows() || cols != y_pred.getCols()) {
		throw std::invalid_argument("Dimensiones incompatibles para gradiente de MSE");
	}
	core::Tensor grad(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			float value = 2.0f * (y_pred.get(i, j) - y_true.get(i, j));
			grad.set(i, j, value);
		}
	}
	return grad;
}
