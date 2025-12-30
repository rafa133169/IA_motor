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

core::Tensor ReLU::gradient(const core::Tensor& tensor) {
    int rows = tensor.getRows();
    int cols = tensor.getCols();
    core::Tensor grad(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            grad.set(i, j, tensor.get(i, j) > 0.0f ? 1.0f : 0.0f);
        }
    }
    return grad;
}
