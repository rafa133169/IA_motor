
# NevronCpp: Motor de Deep Learning de Alto Rendimiento desde Cero

> **Estado:** En Desarrollo (Arquitectura Modular y Polimórfica)
> **Lenguaje:** C++17 (Standard Library Only)
> **Dependencias:** Ninguna (Zero-Dependency)

---

## Sobre el Proyecto

**NevronCpp** es un motor educativo de Inteligencia Artificial en C++ puro, diseñado para máxima eficiencia y extensibilidad. Su arquitectura modular permite experimentar con nuevas capas, funciones de activación y pérdidas, facilitando la investigación y el aprendizaje profundo de los fundamentos de IA.

### Características Clave
- **Gestión Manual de Memoria:** Uso de punteros y referencias para evitar copias innecesarias.
- **Flat Buffers:** Representación de tensores como un único vector lineal para maximizar la localidad de caché y el rendimiento.
- **Operaciones In-Place:** Modificación directa de los datos para ahorrar memoria.
- **Arquitectura Polimórfica:** Interfaces base para capas (`Layer`), funciones de activación (`Activation`) y pérdidas (`Loss`), permitiendo composición y extensión sencilla.

---

## Filosofía Técnica

### Arquitectura de Memoria Flat Buffer
En vez de vectores anidados (`std::vector<std::vector<float>>`), se utiliza un único arreglo lineal (`std::vector<float>`) para representar matrices N-dimensionales:

$$
	ext{Tensor} = [x_1, x_2, \ldots, x_{n \cdot m}]
$$

**Ventajas:**
- Datos contiguos en memoria
- Prefetching eficiente en caché L1/L2
- Acceso ultra rápido en operaciones matemáticas

### Aritmética de Punteros
Para activaciones y transformaciones, se prefiere la iteración de punteros (`*ptr++`) sobre la indexación tradicional `[i][j]`, minimizando la sobrecarga de cálculo de direcciones.

---

## Fundamentos Matemáticos


### Producto Punto (Capa Densa)
Multiplicación de matrices $A \in \mathbb{R}^{m \times n}$ y $B \in \mathbb{R}^{n \times p}$:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

*Implementación:* Algoritmo naive $O(n^3)$ optimizado por acceso lineal a memoria.

### Suma y Operaciones Elementwise
Para dos tensores $X, Y \in \mathbb{R}^{m \times n}$:

$$
Z = X + Y \implies z_{ij} = x_{ij} + y_{ij}
$$

$$
Z = X \odot Y \implies z_{ij} = x_{ij} \cdot y_{ij}
$$

### Transposición de Matriz
Dada $A \in \mathbb{R}^{m \times n}$:

$$
A^T_{ij} = A_{ji}
$$

### Norma Euclidiana (L2)
Para $x \in \mathbb{R}^n$:

$$
\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}
$$

### Función de Activación: ReLU
Rectified Linear Unit introduce no-linealidad:

$$
f(x) = \max(0, x)
$$

*Optimización:* Se aplica **in-place** sobre el tensor, sobrescribiendo la memoria original.

**Gradiente de ReLU:**
$$
f'(x) = \begin{cases}
1 & x > 0 \\
0 & x \leq 0
\end{cases}
$$

### Función de Activación: Sigmoide

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Gradiente de Sigmoide:**
$$
\sigma'(x) = \sigma(x) (1 - \sigma(x))
$$

### Función de Activación: Tanh

$$
	anh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

**Gradiente de Tanh:**
$$
\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)
$$

### Función de Coste: MSE (Mean Squared Error)
Evalúa la precisión del modelo:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Gradiente de MSE respecto a la predicción:**
$$
\frac{\partial MSE}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

### Derivada Parcial y Regla de la Cadena (Backpropagation)
Para una función compuesta $L(y(x))$:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

### Softmax (Clasificación Multiclase)

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}
$$

**Gradiente de Softmax:**
$$
\frac{\partial \mathrm{softmax}(x)_i}{\partial x_j} = \mathrm{softmax}(x)_i (\delta_{ij} - \mathrm{softmax}(x)_j)
$$

---

## Arquitectura Modular y Polimórfica

### Interfaces Base

```cpp
class Layer {
public:
    virtual ~Layer() = default;
    virtual core::Tensor forward(const core::Tensor& input) = 0;
    virtual core::Tensor backward(const core::Tensor& grad_output) = 0;
};

class Activation {
public:
    virtual ~Activation() = default;
    virtual void apply_inplace(core::Tensor& tensor) = 0;
    virtual core::Tensor gradient(const core::Tensor& tensor) = 0;
};

class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute(const core::Tensor& y_true, const core::Tensor& y_pred) = 0;
    virtual core::Tensor gradient(const core::Tensor& y_true, const core::Tensor& y_pred) = 0;
};
```

### Ejemplo de Uso Polimórfico

```cpp
#include <memory>
std::unique_ptr<Activation> activation = std::make_unique<layers::ReLU>();
std::unique_ptr<Loss> loss = std::make_unique<losses::MSE>();
activation->apply_inplace(tensor);
float mse = loss->compute(y_true, tensor);
core::Tensor grad = loss->gradient(y_true, tensor);
```

---

## Ejemplo de Flujo de Entrenamiento

```cpp
for (int epoch = 0; epoch < epochs; ++epoch) {
    // Forward
    core::Tensor outputs = inputs.dot(weights);
    activation->apply_inplace(outputs);
    // Loss
    float loss_value = loss->compute(targets, outputs);
    // Gradiente
    core::Tensor grad_loss = loss->gradient(targets, outputs);
    // Actualización de pesos (simplificada)
    // ...
}
```

---

## Pruebas y Ejemplos

- Pruebas unitarias con GoogleTest para Tensor, ReLU y MSE.
- Ejemplos de uso básico, entrenamiento y polimorfismo en la carpeta `examples/`.

---

## Contribuciones

El proyecto está abierto a mejoras, nuevas capas, funciones de activación y pérdidas. ¡Explora, aprende y contribuye!
