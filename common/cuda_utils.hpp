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

// Crashes if the given CUDA error status is not successful
//
// Parameters:
//  - status: the status returned from a CUDA call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cuda(cudaError_t status, std::string message) {
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status);
    std::cerr << ". " << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Crashes if the given cuBLAS error status is not successful
//
// Parameters:
//  - status: the status returned from a cuBLAS call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cublas(cublasStatus_t status, std::string message) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error: " << cublasGetStatusString(status);
    std::cerr << ". " << message << std::endl;
    exit(EXIT_FAILURE);
  }
}


#endif // CUDA_UTILS_H_