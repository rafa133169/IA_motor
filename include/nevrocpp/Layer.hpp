#pragma once
#include "core/Tensor.hpp"

namespace nevrocpp {

class Layer {
public:
    virtual ~Layer() = default;
    virtual core::Tensor forward(const core::Tensor& input) = 0;
    virtual core::Tensor backward(const core::Tensor& grad_output) = 0;
};

} // namespace nevrocpp
