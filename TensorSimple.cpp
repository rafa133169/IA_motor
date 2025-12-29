#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip> // Para imprimir bonito

class TensorSimple {
private:
    // PROPIEDADES
    int rows;
    int cols;
    
    // ALMACENAMIENTO
    // Usamos un solo vector (1D) para representar la matriz (2D).
    // Esto es mucho más rápido que vector<vector<float>>.
    std::vector<float> data;

public:
    // CONSTRUCTOR
    TensorSimple(int r, int c) : rows(r), cols(c) {
        // Reservamos memoria contigua
        data.resize(rows * cols, 0.0f); 
    }

    // ACCESO A DATOS (Setter)
    // Mapeamos coordenada (fil, col) a índice lineal: index = f * num_cols + c
    void set(int r, int c, float value) {
        if (r >= rows || c >= cols) throw std::out_of_range("Indice fuera de rango");
        data[r * cols + c] = value;
    }

    // ACCESO A DATOS (Getter)
    float get(int r, int c) const {
        if (r >= rows || c >= cols) throw std::out_of_range("Indice fuera de rango");
        return data[r * cols + c];
    }

    // OBTENER DIMENSIONES
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    // ---------------------------------------------------------
    // EL CORAZÓN DE LA IA: MULTIPLICACIÓN DE MATRICES (Dot Product)
    // ---------------------------------------------------------
    // Realiza la operación: C = A * B
    // Si A es (m x n) y B es (n x p), el resultado C es (m x p)
    TensorSimple dot(const TensorSimple& other) const {
        // 1. Validación de dimensiones (Regla del álgebra lineal)
        if (this->cols != other.rows) {
            throw std::invalid_argument("Dimensiones incompatibles para multiplicacion.");
        }

        int m = this->rows;
        int n = this->cols; // o other.rows
        int p = other.cols;

        // 2. Crear tensor resultante
        TensorSimple result(m, p);

        // 3. Algoritmo de multiplicación (El triple loop clásico)
        // Ojo: Esto es O(n^3). Librerías como Eigen optimizan esto brutalmente,
        // pero para aprender, esta es la lógica pura.
        for (int i = 0; i < m; ++i) {           // Recorre filas de A
            for (int j = 0; j < p; ++j) {       // Recorre columnas de B
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {   // Producto punto
                    sum += this->get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    // UTILIDAD: Imprimir matriz
    void print() const {
        for (int i = 0; i < rows; ++i) {
            std::cout << "| ";
            for (int j = 0; j < cols; ++j) {
                std::cout << std::fixed << std::setprecision(2) << get(i, j) << " ";
            }
            std::cout << "|" << std::endl;
        }
    }
};

int main() {
    // EJEMPLO DE USO: SIMULANDO UNA CAPA DE RED NEURONAL
    
    // 1. INPUT (Digamos, datos de 2 muestras con 3 características cada una) -> Matriz 2x3
    TensorSimple inputs(2, 3);
    inputs.set(0, 0, 1.0); inputs.set(0, 1, 2.0); inputs.set(0, 2, 3.0);
    inputs.set(1, 0, 4.0); inputs.set(1, 1, 5.0); inputs.set(1, 2, 6.0);

    std::cout << "--- Inputs (2x3) ---" << std::endl;
    inputs.print();

    // 2. PESOS (Weights) que conectan 3 entradas a 2 neuronas de salida -> Matriz 3x2
    TensorSimple weights(3, 2);
    weights.set(0, 0, 0.5); weights.set(0, 1, 0.1);
    weights.set(1, 0, 0.2); weights.set(1, 1, 0.4);
    weights.set(2, 0, 0.1); weights.set(2, 1, 0.8);

    std::cout << "\n--- Pesos (3x2) ---" << std::endl;
    weights.print();

    // 3. FORWARD PASS (Multiplicación)
    // Resultado esperado: Matriz 2x2
    try {
        TensorSimple output = inputs.dot(weights);
        std::cout << "\n--- Output (Forward Pass) (2x2) ---" << std::endl;
        output.print();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}