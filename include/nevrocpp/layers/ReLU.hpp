#pragma once
#include "nevrocpp/core/Tensor.hpp"
#include "nevrocpp/Activation.hpp"

namespace nevrocpp::layers {

class ReLU : public Activation {
public:
    // Aplica ReLU directamente sobre el tensor
    void apply_inplace(core::Tensor& tensor) override;
    core::Tensor gradient(const core::Tensor& tensor) override;
};

}
