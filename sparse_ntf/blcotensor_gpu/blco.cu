#include <iostream>
#include "blco.h"
#include "cuda_utils.h"

// #include "blco_tensor.hpp"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <chrono>


#define MAX_NUM_MODES 5

__constant__ _IType MASKS[MAX_NUM_MODES];
__constant__  int POS[MAX_NUM_MODES];

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

MAT_GPU * send_mat_to_gpu(MAT * mat) {
  
  int num_elements = mat->n_rows * mat->n_cols;
  
  MAT_GPU * _mat = new MAT_GPU(mat->n_rows, mat->n_cols);
  
  check_cuda(cudaMallocHost(&_mat->vals, sizeof(double) * num_elements), "cudaMalloc mat to gpu");
  check_cuda(cudaMemcpy(_mat->vals, mat->memptr(), sizeof(double) * num_elements, cudaMemcpyHostToDevice), "cudaMemcpy mat to gpu");
  return _mat;
};

MAT_GPU ** send_mats_to_gpu(MAT * mats, int num_modes) {
  MAT_GPU ** _mats_gpu = new MAT_GPU*[num_modes];
  for (int m = 0; m < num_modes; ++m) {
    _mats_gpu[m] = send_mat_to_gpu(&mats[m]);
  }
  return _mats_gpu;
};

// Function to free GPU memory for a MAT_GPU object
void free_mat_on_gpu(MAT_GPU * mat_gpu) {
    cudaFree(mat_gpu->vals);  // Free the GPU data memory
    delete mat_gpu;           // Delete the MAT_GPU object
}

void free_mats_on_gpu(MAT_GPU ** mats_gpu, int num_modes) {
    for (int m = 0; m < num_modes; ++m) {
        // Assuming you have a function to free GPU memory for MAT_GPU
        free_mat_on_gpu(mats_gpu[m]);
    }
    delete[] mats_gpu;
}


BLCOTensorGPU * copy_blcotensor_to_device(
  _IType max_block_size,
  int num_modes,
  unsigned long long * dimensions,
  _IType numels,
  _IType * mode_mask,
  int * mode_pos,
  _IType block_count,
  BLCOBlock ** blocks
  ) {
    BLCOTensorGPU * bt = new BLCOTensorGPU;
    bt->m_modes = num_modes;
    bt->m_blco_mode_mask = make_device_copy(mode_mask, num_modes, "cudaMemcpy mmakeode_masks");
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
      printf("mode_mask[%d]: %llx, mode_pos: %d\n", m, mode_mask[m], mode_pos[m]);
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

    cudaMemcpyToSymbol(MASKS, mode_mask, num_modes * sizeof(_IType));
    cudaMemcpyToSymbol(POS, mode_pos, num_modes * sizeof(int));
    printf("===== BLCO mask sent to gpu =====\n");

    // if (do_batching) bt->warp_info_gpu
    return bt;
};

void send_blco_to_gpu(){
  
};

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
