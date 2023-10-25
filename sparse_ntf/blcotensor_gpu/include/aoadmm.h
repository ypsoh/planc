#ifndef AOADMM_GPU_H_
#define AOADMM_GPU_H_

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "vector_ops.h"
#include "cuda_utils.h"
#include "blco.h"
#include <iostream>

void aoadmm_update(MAT_GPU * fm, MAT_GPU * aux_fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram, int admm_iter, double tolerance);

#endif // AOADMM_GPU_H_