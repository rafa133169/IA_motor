#include <iostream>
#include "nevrocpp/core/Tensor.hpp"
#include "nevrocpp/losses/MSE.hpp"
#include "nevrocpp/layers/ReLU.hpp"

using namespace nevrocpp;

// Simula un paso de entrenamiento para una sola capa densa (sin bias)
void train_step(core::Tensor& inputs, core::Tensor& weights, const core::Tensor& targets, float lr) {
    // Forward: salida = inputs.dot(weights)
    core::Tensor outputs = inputs.dot(weights);
    // Activación
    layers::ReLU relu;
    relu.apply_inplace(outputs);
    // Cálculo de error
    losses::MSE mse;
    float loss = mse.compute(targets, outputs);
    std::cout << "Loss: " << loss << std::endl;
    // Gradiente del error respecto a la salida
    core::Tensor grad_loss = mse.gradient(targets, outputs);
    // Gradiente respecto a los pesos (simplificado, sin backprop completo)
    // dL/dW = X^T . grad_loss
    core::Tensor inputs_T(inputs.getCols(), inputs.getRows());
    for (int i = 0; i < inputs.getRows(); ++i)
        for (int j = 0; j < inputs.getCols(); ++j)
            inputs_T.set(j, i, inputs.get(i, j));
    core::Tensor grad_weights = inputs_T.dot(grad_loss);
    // Actualización de pesos: W = W - lr * grad_weights
    for (int i = 0; i < weights.getRows(); ++i)
        for (int j = 0; j < weights.getCols(); ++j)
            weights.set(i, j, weights.get(i, j) - lr * grad_weights.get(i, j));
}

int main() {
    // Datos de ejemplo
    core::Tensor inputs(2, 3); // 2 muestras, 3 features
    inputs.set(0, 0, 1.0f); inputs.set(0, 1, 2.0f); inputs.set(0, 2, 3.0f);
    inputs.set(1, 0, 4.0f); inputs.set(1, 1, 5.0f); inputs.set(1, 2, 6.0f);
    core::Tensor weights(3, 1); // 3 features, 1 salida
    weights.set(0, 0, 0.1f); weights.set(1, 0, 0.2f); weights.set(2, 0, 0.3f);
    core::Tensor targets(2, 1); // 2 muestras, 1 salida
    targets.set(0, 0, 1.0f); targets.set(1, 0, 2.0f);
    float lr = 0.01f;
    for (int epoch = 0; epoch < 10; ++epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        train_step(inputs, weights, targets, lr);
    }
    std::cout << "Pesos finales:" << std::endl;
    weights.print();
    return 0;
}
