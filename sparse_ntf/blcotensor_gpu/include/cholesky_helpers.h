#ifndef CHOLESKY_HELPERS_H_
#define CHOLESKY_HELPERS_H_

#include <iostream>
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cassert"

#include "config.h"
#include "cuda_utils.h"


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


// cusolver API error checking
#define CHECK_CUSOLVER(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

const int EXIT_UNSUPPORTED = 2;

/**
 * Create a batch (num_copys) of square matrices given mat_to_copy
*/
void create_batch_sqmatrices(double * mat_to_copy, double * batch_mats, size_t num_copys, size_t dims);
void apply_mask_to_batch_sqmatrices(double * batch_sqmats, int dims, int num_cols, unsigned int * masks);
void apply_uc_mask_to_batch_sqmatrices(double * batch_sqmats, int dims, int num_cols, unsigned char * masks);
int cusolverBatchedCholesky(double * matA_vals, double * matB_vals, uint block_size, uint num_blocks);

void idxBasedCopy(double * dst, double * src, unsigned int * ind, int n_rows, int n_cols);
void ucidxBasedCopy(double * dst, double * src, unsigned char * ind, int n_rows, int n_cols);

#endif