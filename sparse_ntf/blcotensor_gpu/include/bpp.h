#ifndef BPP_H_
#define BPP_H_

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

#define CHECK_CUBLAS(func)                                                       \
{                                                                              \
    cublasStatus_t status = (func);                                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                                               \
        std::printf("CUBLAS API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cublasGetStatusString(status), status);                  \
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


void init_passive_non_opt_infea_sets(unsigned char * pset_vals, 
    unsigned char * non_opt_set_vals, unsigned char * infeas_set_vals, double * d_X, double * d_Y, unsigned int size);

void init_alpha_beta_vec(unsigned char * num_trial, unsigned int * num_infeas_val, int nrhs, int rank, unsigned char default_num_trials);
void compute_two_ucmat_column_sum(unsigned char * ucmat1, unsigned char * ucmat2, 
    unsigned int * col_sum_vec, int rank, int nrhs);

void compute_uivec_sum(unsigned int * col_vec, unsigned int * sum_col_vec, int vec_size);
void update_case_vec(unsigned int * non_opt_cols, unsigned char * case_ind_vec, unsigned char * num_trial_val, unsigned int * num_infeas_val, int size);

void update_partition_p1(unsigned int * non_opt_cols, unsigned char * case_ind_vec, unsigned char * num_trial_val, unsigned int * num_infeas_val, unsigned int nrhs);
void update_partition_p2(unsigned char * pset_vals, unsigned char * non_opt_vals, unsigned char * infeas_set_vals, unsigned char * case_ind_vec, unsigned int rank, unsigned int nrhs);
#endif