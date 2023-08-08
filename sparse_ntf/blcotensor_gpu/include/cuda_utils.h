#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <iostream>

#define WARP_SIZE 32
#define TILE_SIZE WARP_SIZE
#define STASH_SIZE 4
#define NUM_STREAMS 8 // set CUDA_DEVICE_MAX_CONNECTIONS env = 1 to 32 (default is 8)
#define LVL3_MAX_MODE_LENGTH 100

#define _IType unsigned long long
#define _FType double
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
T* make_device_copy(T* vector, _IType n, std::string name);

#endif // CUDA_UTILS_H_