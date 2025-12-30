#pragma once
#include "core/Tensor.hpp"

namespace nevrocpp {

class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute(const core::Tensor& y_true, const core::Tensor& y_pred) = 0;
    virtual core::Tensor gradient(const core::Tensor& y_true, const core::Tensor& y_pred) = 0;
};

} // namespace nevrocpp
