#include <iostream>
#include "blco.h"
#include "cuda_utils.h"
#include <cooperative_groups.h>
#include <chrono>

BLCOBlock * generate_block_host(_IType N, _IType nnz) {
  BLCOBlock * b = new BLCOBlock;
  b->block_coords = (_IType *) malloc(sizeof(_IType) * N);
  check_cuda(cudaMallocHost(
    &b->block_coords, sizeof(_IType) * N), "cudaMallocHost block_coords");
  b->m_modes = N;
  b->m_numel = nnz;
  return b;
}

int _hello()
{
    // Launch the CUDA kernel with a single thread block containing a single thread
    // helloGPU<<<1, 4>>>();

    // Wait for the GPU to finish executing
    cudaDeviceSynchronize();

    // Check if any errors occurred during the GPU execution
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaError) << std::endl;
        return 1;
    }

    return 0;
}
