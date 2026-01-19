#include "tensor.h"
#include <cmath>
#include <stdexcept> 
#include <algorithm>

// define matrix multiplication
// returns a new Tensor C = A @ B
Tensor matmul(const Tensor& A, const Tensor& B) {

    // [K] @ [K, N] -> [N]
    // really it's (1, K) x (K, N) = (1, N), but these are tensors, so first vector is literally 1 dimensional
    if (A.shape.size() == 1 && B.shape.size() == 2) {
        int K = A.shape[0];
        int N = B.shape[1];
        if (K != B.shape[0]) {
            throw std::invalid_argument( "Dimension mismatch! A " + shape_str(A) + " B " + shape_str(B));
        }

        // GEMV loop
        // initialize output tensor
        Tensor C({N});

        // size of tile
        const int BN = 64;

        // per-tile loop, which is why we increment by tile size
        // e.g., without min, for a vector of length 100, we'd try to increment up to 128, which is OOB
        for (int j0 = 0; j0 < N; j0 += BN) {

            // take min in case our vector length is not a multiple of the tile size
            int j_max = std::min(j0 + BN, N);

            // initialize output tile
            for (int j = j0; j < j_max; ++j) {
                C.data[j] = 0.0f; 
            }

            // temporal reuse: A's entries show up repeatedly, so 
            // we start by loading one and using it many times
            for (int k = 0; k < K; ++k) {
                const float a = A.data[k];

                // get the address of the B row tile we're using
                // we've gotten through k rows of size N, and we're on the j0th tile of that row
                const float* b_row_tile = &B.data[k * N + j0];

                //contiguous access: consume cache lines of B efficiently
                for (int j = j0; j < j_max; ++j) {
                    C.data[j] += a * b_row_tile[j - j0]; // start at B row tile 0 and increment to max
                }
            }
        }
        return C;
    }
    
    // GEMM loop

    if (A.shape.size() == 2 && B.shape.size() == 2) { // a 2D tensor is a matrix. A.shape will return the dimension of A (e.g., 3x4 = (3, 4))

        // make sure 2x2 matmul is valid
        if (A.shape[1] != B.shape[0]) {
            std::cout << "A.shape[1]: " << A.shape[1] << ", A.shape[0]: " << A.shape[0] << std::endl;
            throw std::invalid_argument( "Dimension mismatch! A " + shape_str(A) + " B " + shape_str(B));
        }

        int M = A.shape[0]; // Rows of A
        int K = A.shape[1]; // Cols of A (and Rows of B)
        int N = B.shape[1]; // Cols of B

        Tensor C({M, N}); // initialize C, an M x N tensor
        std::fill(C.data.begin(), C.data.end(), 0.0f);

        // block sizes
        const int BM = 64;
        const int BN = 64;
        const int BK = 64;

        // matrices are stored row-major
        // A_{i,k} := A[i*K + k]

        // index for rows of A/C
        for (int i0 = 0; i0 < M; i0 += BM) {
            int i_max = std::min(i0 + BM, M);

            // index for columns of B/C
            // having this before k0 means we try to keep C in memory 
            // while 
            for (int j0 = 0; j0 < N; j0 += BN) {
                int j_max = std::min(j0 + BN, N);

                // block index
                // sets the row / column index of a block within A and B
                for (int k0 = 0; k0 < K; k0 += BK) {
                    int k_max = std::min(k0 + BK, K);

                    // iterate over rows of C
                    // in chunks of 64
                    for (int i = i0; i < i_max; ++i) {
                        float* c_row = C.data.data() + i * N + j0;
                        // j0 moves first, so we fix a row and increase the column
                        
                        // iterate over the elements of A / corresponding columns of B
                        // in chunks of 64
                        for (int k = k0; k < k_max; ++k) {

                            const float a = A.data[i * K + k];
                            // as k increases, move along the row of A
                            // as i increases, move to a new row
                            
                            const float* b_row = B.data.data() + k * N + j0;
                            // as j0 increases, move along the row of B
                            // as k increases, move along the column
                            // k moves first, so we fix a column and increase the row
                            // go for BK = 64 rows, then new column

                            // use a scalar in the stored row of A 
                            // while we move through a row of B and C
                            for (int j = j0; j < j_max; ++j) {
                                c_row[j - j0] += a * b_row[j - j0];
                            }
                        }
                    }
                }
            }
        }

        return C;
    } else { throw std::invalid_argument("Matmul only supports 1D/2D tensors for now."); }
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