#pragma once

#include "nevrocpp/core/Tensor.hpp"

namespace nevrocpp::layers {

class ReLU {
public:
    // Aplica ReLU directamente sobre el tensor
    static void apply_inplace(core::Tensor& tensor);
};

}
