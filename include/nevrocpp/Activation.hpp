#pragma once
#include "core/Tensor.hpp"

namespace nevrocpp {

class Activation {
public:
    virtual ~Activation() = default;
    virtual void apply_inplace(core::Tensor& tensor) = 0;
    virtual core::Tensor gradient(const core::Tensor& tensor) = 0;
};

} // namespace nevrocpp
