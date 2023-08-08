#include <iostream>
#include "blco.h"
#include "cuda_utils.h"
// #include "blco_tensor.hpp"
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

BLCOBlock * generate_block_device(_IType N, _IType nnz) {
  BLCOBlock * b = new BLCOBlock;
  b->m_modes = N;
  check_cuda(cudaMalloc(&b->block_coords, sizeof(_IType) * N), "cudaMalloc block_coords");
  
  b->m_numel = nnz;
  check_cuda(cudaMalloc(&b->idx, sizeof(_IType) * nnz), "cudaMalloc block idx");
  check_cuda(cudaMalloc(&b->vals, sizeof(_FType) * nnz), "cudaMalloc block vals");

  return b;
}

BLCOTensorDev * copy_blcotensor_to_device(
  _IType max_block_size,
  int num_modes,
  unsigned int * dimensions,
  _IType numels,
  _IType * mode_mask,
  int * mode_pos,
  _IType block_count,
  BLCOBlock ** blocks
  ) {
    BLCOTensorDev * bt = new BLCOTensorDev;
    bt->m_modes = num_modes;
    bt->m_blco_mode_mask = make_device_copy(mode_mask, num_modes, "cudaMemcpy mode_masks");
    bt->dims = make_device_copy(dimensions, num_modes, "cudaMemcpy dimensions");
    
    bt->m_blco_mode_pos = make_device_copy(mode_pos, num_modes, "cudaMemcpy mode_pos");
    
    bt->m_numel = numels;
    bt->m_num_blocks = block_count;
    bt->m_blocks = new BLCOBlock*[block_count];
    // bt->m_blocks_dev_staging -- no need yet
    // bt->m_blocks_dev_ptr
    // bt->streams
    printf("====== BLCO_GPU Info ======\n");
    printf("nnz: %d, num_blocks: %d, num_modes: %d\n",
      bt->m_numel, bt->m_num_blocks, bt->m_modes);
    for (int m = 0; m < num_modes; ++m) {
      printf("mode_mask[%d]: %llx, mode_pos: %d\n", bt->m_blco_mode_mask[m], bt->m_blco_mode_pos[m]);
      printf("bt->dims[%d]: %d\n", m, bt->dims[m]);
    }

    // what is this for?
    bool do_batching = false;
    if (do_batching) {
      // set up warp stuff
    } else {
      bt->warp_info_length = 0;
      bt->warp_info = nullptr;
      bt->warp_info_gpu = nullptr;
    }

    // move blocks to gpu and generate warp info
    for (int i = 0; i < block_count; ++i) {
      // max block size
      _IType mb = (max_block_size == 0) ? blocks[i]->m_numel : max_block_size;
      bt->m_blocks[i] = generate_block_device(bt->m_modes, mb);
      // if batching create warp info for blco tesor 
    }
    printf("====== Created BLCOBLocks ======\n");

    // if (do_batching) bt->warp_info_gpu
    return bt;
};

void send_blco_to_gpu(){

};

void send_masks_to_gpu(){};
void allocate_gpu_mem(){};
void send_factors_to_gpu(){};

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
