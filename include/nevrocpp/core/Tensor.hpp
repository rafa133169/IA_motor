#pragma once

#include <vector>

namespace nevrocpp::core {

class Tensor {
private:
    int rows;
    int cols;
    std::vector<float> data;

public:
    // Constructor
    Tensor(int r, int c);

    // Acceso a datos
    void set(int r, int c, float value);
    float get(int r, int c) const;

    // Dimensiones
    int getRows() const;
    int getCols() const;

    // Operaciones matemáticas
    Tensor dot(const Tensor& other) const;

    // Utilidades
    void print() const;

    // Acceso directo a memoria (futuras optimizaciones)
    float* data_ptr();
};

} // namespace nevrocpp::core
