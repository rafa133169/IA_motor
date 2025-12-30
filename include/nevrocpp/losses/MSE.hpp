// Archivo: include/nevrocpp/losses/MSE.hpp
#pragma once
#include "nevrocpp/core/Tensor.hpp"
#include "nevrocpp/Loss.hpp"

namespace nevrocpp::losses {

class MSE : public nevrocpp::Loss {
public:
	// Calcula el error cuadrático medio entre dos tensores
	float compute(const core::Tensor& y_true, const core::Tensor& y_pred) override;
	core::Tensor gradient(const core::Tensor& y_true, const core::Tensor& y_pred) override;
};

} // namespace nevrocpp::losses
