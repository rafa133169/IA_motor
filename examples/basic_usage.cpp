#include <iostream>
#include "nevrocpp/core/Tensor.hpp"

using nevrocpp::core::Tensor;

int main() {

    Tensor inputs(2, 3);
    inputs.set(0, 0, 1.0);
    inputs.set(0, 1, 2.0);
    inputs.set(0, 2, 3.0);
    inputs.set(1, 0, 4.0);
    inputs.set(1, 1, 5.0);
    inputs.set(1, 2, 6.0);

    std::cout << "--- Inputs ---\n";
    inputs.print();

    Tensor weights(3, 2);
    weights.set(0, 0, 0.5);
    weights.set(0, 1, 0.1);
    weights.set(1, 0, 0.2);
    weights.set(1, 1, 0.4);
    weights.set(2, 0, 0.1);
    weights.set(2, 1, 0.8);

    std::cout << "\n--- Weights ---\n";
    weights.print();

    Tensor output = inputs.dot(weights);

    std::cout << "\n--- Output ---\n";
    output.print();

    return 0;
}
