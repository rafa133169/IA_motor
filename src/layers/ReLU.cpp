#include "nevrocpp/layers/ReLU.hpp"

namespace nevrocpp::layers {

void ReLU::apply_inplace(core::Tensor& tensor) {
    float* ptr = tensor.data_ptr();
    int size = tensor.getRows() * tensor.getCols();

    // Loop ultra cache-friendly
    for (int i = 0; i < size; ++i) {
        ptr[i] = (ptr[i] > 0.0f) ? ptr[i] : 0.0f;
    }
}

}
