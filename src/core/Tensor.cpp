#include "nevrocpp/core/Tensor.hpp"

#include <iostream>
#include <stdexcept>
#include <iomanip>

namespace nevrocpp::core {

Tensor::Tensor(int r, int c) : rows(r), cols(c) {
    data.resize(rows * cols, 0.0f);
}

void Tensor::set(int r, int c, float value) {
    if (r >= rows || c >= cols)
        throw std::out_of_range("Indice fuera de rango");
    data[r * cols + c] = value;
}

float Tensor::get(int r, int c) const {
    if (r >= rows || c >= cols)
        throw std::out_of_range("Indice fuera de rango");
    return data[r * cols + c];
}

int Tensor::getRows() const { return rows; }
int Tensor::getCols() const { return cols; }

Tensor Tensor::dot(const Tensor& other) const {
    if (cols != other.rows)
        throw std::invalid_argument("Dimensiones incompatibles");

    Tensor result(rows, other.cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols; ++k) {
                sum += get(i, k) * other.get(k, j);
            }
            result.set(i, j, sum);
        }
    }

    return result;
}

void Tensor::print() const {
    for (int i = 0; i < rows; ++i) {
        std::cout << "| ";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(2)
                      << get(i, j) << " ";
        }
        std::cout << "|\n";
    }
}

float* Tensor::data_ptr() {
    return data.data();
}

} // namespace nevrocpp::core
