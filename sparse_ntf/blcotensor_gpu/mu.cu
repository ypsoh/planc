#include "mu.h"
#include <cmath>

__global__ void update_fm(double * h, double * o_mttkrp_vals, double * temp, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    h[idx] = h[idx] * o_mttkrp_vals[idx] / temp[idx];
  }
}

/* 4x3
0: (0, 0) 4: (0, 1) 8: (0, 2) 
1: (1, 0) 5: (1, 1) 9: (1, 2) 
2: (2, 0) 6: (2, 1) 10: (2, 2) 
3: (3, 0) 7: (3, 1) 11: (3, 2) 
*/
// 12
/*
0: (0, 0) 3: (0, 1) 6: (0, 2) 9: (0, 3)
1: (1, 0) 4: (1, 1) 7: (1, 2) 10: (1, 3)
2: (2, 0) 5: (2, 1) 8: (2, 2) 11: (2, 3)
*/

// 0 -> 0 - 0 * 3 + 0
// 1 -> 3 - 1 * 3 + 0
// 2 -> 6 - 2 * 3 + 0
// 3 -> 9 - 3 * 3 + 0

// 4 -> 1 - 0 * 3 + 1
// 5 -> 4 - 1 * 3 + 1
// 6 -> 7 - 2 * 3 + 1
// 7 -> 10 - 3 * 3 + 1

// 8 -> 2 - 0 * 3 + 2
// 9 -> 5
// 10 -> 8
// 11 -> 11 - 3 * 3 + 2

// 0 -> 0 -
// 1 -> 4 - (idx % n)* m + idx / n
// 2 -> 8

// 3 -> 1
// 4 -> 5
// 5 -> 9 

// 6 -> 2 
// 7 -> 6
// 8 -> 10


void mu_update(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram) {
  // need matrix multiplicatin -- fm * gram
  int m = fm->n_rows; // rows in A and C
  int n = gram->n_cols; // cols in B and C
  int k = fm->n_cols; // cols in A and C

  // create temp matrix
  double * d_temp, * d_mttkrp_t;
  check_cuda(
    cudaMalloc((void**)&d_temp, m * n * sizeof(double)),
    "malloc result matrix"
  );
  check_cuda(
    cudaMalloc((void**)&d_mttkrp_t, m * n * sizeof(double)),
    "malloc mttkrp_t matrix"
  );

  value_fill(d_temp, m * n, EPSILON);
  check_cuda(cudaDeviceSynchronize(), "value fill -- epsilon");

  mat_mat_mul(fm->vals, gram->vals, d_temp, m, n, k, 1.0, 1.0);

  // Update factor matrix
  int num_elements = m * k;
  int num_blocks = (num_elements + TILE_SIZE - 1) / TILE_SIZE;

  transpose_mat<<<num_blocks, TILE_SIZE>>>(d_mttkrp_t, o_mttkrp_gpu->vals, k, m);
  check_cuda(cudaDeviceSynchronize(), "sync after normaliztation");
  // check_cuda(cudaMemcpy(o_mttkrp_gpu->vals, d_mttkrp_t, m * k * sizeof(double), cudaMemcpyDeviceToDevice), "copy o_mttkrp_t to o_mttkrp");

  update_fm<<<num_blocks, TILE_SIZE>>>(fm->vals, d_mttkrp_t, d_temp, num_elements);
  cudaFree(d_temp);
  cudaFree(d_mttkrp_t);
}
