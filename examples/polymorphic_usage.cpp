#include <iostream>
#include <memory>
#include "nevrocpp/core/Tensor.hpp"
#include "nevrocpp/losses/MSE.hpp"
#include "nevrocpp/layers/ReLU.hpp"
#include "nevrocpp/Activation.hpp"
#include "nevrocpp/Loss.hpp"

using namespace nevrocpp;

int main() {
    // Uso polimórfico de Activation y Loss
    std::unique_ptr<Activation> activation = std::make_unique<layers::ReLU>();
    std::unique_ptr<Loss> loss = std::make_unique<losses::MSE>();

    core::Tensor inputs(1, 3);
    inputs.set(0, 0, -1.0f); inputs.set(0, 1, 2.0f); inputs.set(0, 2, -3.0f);
    std::cout << "Input: "; inputs.print();

    activation->apply_inplace(inputs);
    std::cout << "After ReLU: "; inputs.print();

    core::Tensor y_true(1, 3);
    y_true.set(0, 0, 0.0f); y_true.set(0, 1, 2.0f); y_true.set(0, 2, 0.0f);

    float mse = loss->compute(y_true, inputs);
    std::cout << "MSE: " << mse << std::endl;

    core::Tensor grad = loss->gradient(y_true, inputs);
    std::cout << "MSE Gradient: ";
    grad.print();

    core::Tensor relu_grad = activation->gradient(inputs);
    std::cout << "ReLU Gradient: ";
    relu_grad.print();

    return 0;
}
