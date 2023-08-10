#ifndef COMMON_BLCO_HPP_
#define COMMON_BLCO_HPP_

#include <iostream>

#include "cuda_utils.h"
#include <cstdint>
#define ARMA_ALLOW_FAKE_GCC // to use arma::mat
#include <armadillo>
// this imports causes following warning
// "WARNING: this compiler is pretending to be GCC but it may not be fully compatible;"
#include "common/utils.h"
// #include <cooperative_groups.h>
// #include <chrono>
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

// almost identical as BLCOTensor but only contains essential stuff for gpu execution
struct BLCOTensorGPU {
  _IType m_modes;
  BLCOBlock ** m_blocks;
  unsigned long long * dims;

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


BLCOTensorGPU * copy_blcotensor_to_device(
  _IType max_block_size,
  int num_modes,
  unsigned long long * dimensions,
  _IType numels,
  _IType * mode_mask,
  int * mode_pos,
  _IType block_count,
  BLCOBlock ** blocks);

BLCOBlock * generate_block_host(_IType N, _IType nnz);

void send_blco_to_gpu();

MAT_GPU * send_mat_to_gpu(MAT * mat);
MAT_GPU ** send_mats_to_gpu(MAT * mats, int num_modes);
void free_mats_on_gpu(MAT_GPU ** mats_gpu, int num_modes);
void free_mat_on_gpu(MAT_GPU * mat_gpu);

void allocate_gpu_mem();

// MAT * send_mat_to_gpu(MAT * mats, int num_modes); // send mats x num_modes
// MAT * send_mat_to_gpu(MAT * mat);

int _hello();

#endif