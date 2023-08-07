#ifndef COMMON_BLCO_HPP_
#define COMMON_BLCO_HPP_

#include <iostream>
// #include "common/utils.h"
// #include "common/ncpfactors.hpp"
// #include "common/ntf_utils.hpp"
// #include "blco_tensor.hpp"
// #include "cuda_utils.h"
// #include <cooperative_groups.h>
// #include "blco_tensor.hpp"
// #include <chrono>
#define _IType unsigned long long


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
BLCOBlock * generate_block_host(_IType N, _IType nnz);
int _hello();

#endif