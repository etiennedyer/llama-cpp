#include "tensor.h"
#include <cmath>     // for sqrt, exp
#include <stdexcept> // for throwing errors

// define matrix multiplication
// Returns a new Tensor C = A @ B
Tensor matmul(const Tensor& A, const Tensor& B) {

    // 1D x 2D: [K] @ [K, N] -> [N]
    if (A.shape.size() == 1 && B.shape.size() == 2) {
        int K = A.shape[0];
        int N = B.shape[1];
        if (K != B.shape[0]) {
            throw std::invalid_argument( "Dimension mismatch! A " + shape_str(A) + " B " + shape_str(B));
        }
        Tensor C({N});
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A.data[k] * B.data[k * N + j];
            }
            C.data[j] = sum;
        }
        return C;
    }
    
    // Validation: Are we multiplying compatible shapes?
    if (A.shape.size() == 2 && B.shape.size() == 2) { // a 2D tensor is a matrix. A.shape will return the dimension of A (e.g., 3x4 = (3, 4))

        // make sure 2x2 matmul is valid
        if (A.shape[1] != B.shape[0]) {
            std::cout << "A.shape[1]: " << A.shape[1] << ", A.shape[0]: " << A.shape[0] << std::endl;
            throw std::invalid_argument( "blahhh Dimension mismatch! A " + shape_str(A) + " B " + shape_str(B));
        }

        int M = A.shape[0]; // Rows of A
        int K = A.shape[1]; // Cols of A (and Rows of B)
        int N = B.shape[1]; // Cols of B

        Tensor C({M, N}); // initialize an M x N tensor named C

        // Naive Triple Loop (O(N^3))
        for (int i = 0; i < M; i++) {           // Iterate over rows of A
            for (int j = 0; j < N; j++) {       // Iterate over cols of B
                
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {   // Dot product reduction
                    // A[i, k] * B[k, j]
                    // We use flattened indices: index = row * total_cols + col
                    sum += A.data[i * K + k] * B.data[k * N + j];
                }
                
                C.data[i * N + j] = sum; // modify the data vector that exists within C
            }
        }

        return C;
    } else {
        throw std::invalid_argument("Matmul only supports 1D/2D tensors for now.");
    }
}

// add B to A
void add_inplace(Tensor& A, const Tensor& B) {
    if (A.shape != B.shape) {
        throw std::invalid_argument("Shape mismatch in add_inplace");
    }

    for (size_t i = 0; i < A.data.size(); ++i) {
        A.data[i] += B.data[i];
    }
}

void silu_inplace(Tensor& A) {
    for (size_t i = 0; i < A.data.size(); ++i) {
        float x = A.data[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        A.data[i] = x * sigmoid;
    }
}

void rms_norm_inplace(Tensor& A, const Tensor& weight, float eps) {
    // We assume A is [Batch, Dim] or [Tokens, Dim]
    // The normalization happens across the last dimension (Dim).
    
    int dim = A.shape.back(); // The hidden size (e.g., 4096)
    int num_rows = A.data.size() / dim; // How many tokens we have

    for (int r = 0; r < num_rows; ++r) {
        float* row_start = &A.data[r * dim]; // point and derefenrence, we don't need a copy because we're just finding the index
        // get the index by multiplying the "pass" we're on by the dimension of the row

        //Calculate Sum of Squares for this row
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum_sq += row_start[i] * row_start[i];
        }

        // Calculate RMS (Root Mean Square)
        float rms = sqrtf(sum_sq / dim + eps);
        float scale = 1.0f / rms; // Optimization: pre-calculate division

        // Normalize and apply learned weight
        for (int i = 0; i < dim; ++i) {
            // formula: (x / rms) * weight
            row_start[i] = (row_start[i] * scale) * weight.data[i];
        }
    }
}

// debug tool, return the tensor's shape as a string
static std::string shape_str(const Tensor& t) {

    std::string s = "";

    for (size_t i = 0; i < t.shape.size(); ++i) {
        s += std::to_string(t.shape[i]);
        if (i + 1 < t.shape.size()) s += ", ";
    }
    return s;
}