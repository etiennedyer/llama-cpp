#pragma once // once directive to prevent multiple inclusions
#include <vector> 
#include <iostream>

struct Tensor { //use a struct instead of a class to have members public by default. could have used class but this is easier to read
    std::vector<float> data; // this will be a 1D array
    std::vector<int> shape; // this tells us how to interpret the array, e.g., (4096, 4096)

    //constructor
    Tensor(std::vector<int> shape_in) : shape(shape_in) { // initializes shape as shape_in, aka member list initialization
        size_t size = 1; // size_t is a large memory allocation for a non-negative integer

        for (int dim : shape) { //for every integer "dim" in the vector shape
            size *= dim; // we multiply all dimensions to know how much memory to allocate
        }   // e.g., a 3x3 image where each pixel has 3 potential colours is defined by (3x3)x3 = 27 numbers

        data.resize(size, 0.0f); //make our data vector the right size, but fill with 0s
    }
};

// Function declarations
// promise these functions will exist somewhere in the project

Tensor matmul(const Tensor& A, const Tensor& B); // returns a tensor, but A and B are const
void add_inplace(Tensor& A, const Tensor& B); // does not return anything, A is modified, B is constant
void silu_inplace(Tensor& A); //does not return anything, A is modified,
void rms_norm_inplace(Tensor& A, const Tensor& weight, float eps = 1e-5f); //does not return anything, A is modified. epsilon will default to this value, but you can manually specify it if you want
static std::string shape_str(const Tensor& t);