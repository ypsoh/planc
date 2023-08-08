#ifndef COMMON_BLCO_HPP_
#define COMMON_BLCO_HPP_

#include <iostream>

// this imports causes following warning
// "WARNING: this compiler is pretending to be GCC but it may not be fully compatible;"
#include "common/utils.h"
#include "cuda_utils.h"
// #include <cooperative_groups.h>
// #include <chrono>
#define _IType unsigned long long
#define _FType double


struct BLCOBlock {
  _IType m_modes;
  _IType m_numel;

  double * vals; // offset pointers into main memory 
  _IType * idx;  // offset pointers into main memory

  // Length `nmodes` array, the coordinates of this block in the BLCO tensor
  // needed to retreive original alto idx and to eventually original index
  unsigned long long * block_coords = nullptr;

  // double** pmatrices_staging_ptr = nullptr;
  // double** pmatrices = nullptr;
};

// almost identical as BLCOTensor but only contains essential stuff for gpu executtion
struct BLCOTensorDev {
  _IType m_modes;
  BLCOBlock ** m_blocks;
  unsigned int * dims;

  _IType m_numel;
  int m_block_count = 0;
  int m_num_blocks = 0;
  int m_max_block_size = 16777216;

  int * m_blco_mode_pos;
  unsigned long long * m_blco_mode_mask;

  cudaStream_t * m_streams;
  _IType warp_info_length = 0;
  _IType * warp_info = nullptr;
  _IType * warp_info_gpu = nullptr;
};

BLCOTensorDev * copy_blcotensor_to_device(
  _IType max_block_size,
  int num_modes,
  unsigned int * dimensions,
  _IType numels,
  const _IType * mode_mask,
  const int * mode_pos,
  _IType block_count,
  const BLCOBlock * const * blocks);

BLCOBlock * generate_block_host(_IType N, _IType nnz);

void send_blco_to_gpu();
void send_masks_to_gpu();
void allocate_gpu_mem();
void send_factors_to_gpu();

int _hello();

#endif