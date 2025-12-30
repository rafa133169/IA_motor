// Archivo: include/nevrocpp/losses/MSE.hpp
#pragma once

#include "nevrocpp/core/Tensor.hpp"

namespace nevrocpp::losses {

class MSE {
public:
	// Calcula el error cuadrático medio entre dos tensores
	static float compute(const core::Tensor& y_true, const core::Tensor& y_pred);

	// Calcula el gradiente de la función MSE respecto a la predicción
	// grad = 2 * (y_pred - y_true)
	static core::Tensor gradient(const core::Tensor& y_true, const core::Tensor& y_pred);
};

} // namespace nevrocpp::losses
