// Archivo: include/nevrocpp/losses/MSE.hpp
#pragma once
#include "nevrocpp/core/Tensor.hpp"
#include "nevrocpp/Loss.hpp"

namespace nevrocpp::losses {

class MSE : public nevrocpp::Loss {
public:
	// Calcula el error cuadrático medio entre dos tensores
	static float compute(const core::Tensor& y_true, const core::Tensor& y_pred);
	static core::Tensor gradient(const core::Tensor& y_true, const core::Tensor& y_pred);

	// Implementación de Loss
	float compute(const core::Tensor& y_true, const core::Tensor& y_pred) override {
		return compute(y_true, y_pred);
	}
	core::Tensor gradient(const core::Tensor& y_true, const core::Tensor& y_pred) override {
		return gradient(y_true, y_pred);
	}
};

} // namespace nevrocpp::losses
