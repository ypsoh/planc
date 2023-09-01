#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "blco.h"
#include <iostream>

#define WARP_SIZE 32
#define TILE_SIZE WARP_SIZE
#define STASH_SIZE 4
#define NUM_STREAMS 8 // set CUDA_DEVICE_MAX_CONNECTIONS env = 1 to 32 (default is 8)
#define LVL3_MAX_MODE_LENGTH 100
#define BLOCK_SIZE 128 // TODO tune?
#define EPSILON 0.000001

#define _IType unsigned long long
#define _FType double

struct MAT_GPU {
  int n_rows;
  int n_cols;
  
  double * vals;
  // constructor
  MAT_GPU(int r, int c) : n_rows(r), n_cols(c) {}
  MAT_GPU(): n_rows(0), n_cols(0) {}
};

// Crashes if the given CUDA error status is not successful
//
// Parameters:
//  - status: the status returned from a CUDA call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cuda(cudaError_t status, std::string message);

// Crashes if the given cuBLAS error status is not successful
//
// Parameters:
//  - status: the status returned from a cuBLAS call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cublas(cublasStatus_t status, std::string message);

template <typename T>
T* make_device_copy(T* vector, _IType n, std::string name) {
    T* d_vector = nullptr;
    check_cuda(cudaMalloc(&d_vector, sizeof(T) * n), "cudaMalloc " + name);
    check_cuda(cudaMemcpy(d_vector, vector, sizeof(T) * n, cudaMemcpyHostToDevice), "cudaMemcpy " + name);
    return d_vector;
};

__global__ void transpose_mat(double * _mat_t, double * _mat, int m, int n);

// Hadamard update, i.e. x <-- x .* y
__global__ void hadamard_kernel(_FType* x, _FType* y, int n);
void vector_hadamard(_FType* x, _FType* y, int n);

// Fills every element of vector x with the same value
__global__ void value_fill_kernel(_FType* x, _IType n, _FType val);
void value_fill(_FType* x, _IType n, _FType val);

void normalize_fm(MAT_GPU * fm, _FType * lambda);

void mat_mat_mul(_FType* a, _FType* b, _FType* c, int m, int n, int k, double alpha = 1.0, double beta = 0.0);
void mat_vec_mul(_FType* a, _FType* b, _FType* c, int m, int n, double alpha = 1.0, double beta = 0.0);

__global__ void vec_add_sub(double* v1, const double* v2, int n, bool add);
__global__ void fix_numerical_error(double* v, int n, const double th, const double repl);

#endif // CUDA_UTILS_H_