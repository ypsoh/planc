#pragma once

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cblas.h"

#include "cuda_runtime.h"
#include "blco.h"
#include "cuda_utils.h"
#include "cassert"

void pseudoinverse_gpu(cusolverDnHandle_t cusolverHandle, cublasHandle_t cublasHandle,
    cudaStream_t stream, double * A, unsigned int n, double * work, unsigned int lwork, int* info, gesvdjInfo_t gesvd_info);
void solveSvdGemm(const MAT_GPU &A, const MAT_GPU &B, MAT_GPU &X);