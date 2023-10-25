#include "matrix_ops.h"

static const double one = 1.0;
static const double zero = 0.0;


// Takes the reciprocal of each vector entry, sets to zero if smaller than tol
__global__ void reciprocal_vector_kernel(double * v, unsigned int n, double tol) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    v[index] = (fabs(v[index]) > tol) ? 1.0 / v[index] : 0;
  }
}

// Use cusolverDnDgesvdj_bufferSize to calculate needed buffer size
void pseudoinverse_gpu(cusolverDnHandle_t cusolverHandle, cublasHandle_t cublasHandle,
    cudaStream_t stream, double* A, unsigned int n, double* work, unsigned int lwork, int* info, gesvdjInfo_t gesvd_info) {

    // I tried Cholesky / QR / LU factorization
    // They scale poorly to larger matrices compared to svd + gemm

    double* U = work;
    double* V = U + n * n;
    double* S = V + n * n;
    work = S + n;
    lwork -= (2 * n * n + n);

    // Gen SVD
    check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
    check_cusolver(cusolverDnDgesvdj(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 
        0, n, n, A, n, S, U, n, V, n, work, lwork, info, gesvd_info), "cusolverDnDgesvdj");

    check_cuda(cudaStreamSynchronize(stream), "cusolverDngesvdj execute");

    // Multiply U by S^-1 (scale rows of U by reciprocal of S);
    unsigned int blocks = n / BLOCK_SIZE + 1;
    double s = 0; // Get largest singular value
    check_cuda(cudaMemcpy(&s, S, sizeof(double), cudaMemcpyDeviceToHost), "memcpy");
    s = n * (nextafter(s, s + 1) - s);
    reciprocal_vector_kernel <<<blocks, BLOCK_SIZE, 0, stream>>>(S, n, s);
    //check_cuda(cudaGetLastError(), "reciprocal_vector launch");
    check_cuda(cudaStreamSynchronize(stream), "reciprocal_vector execute");

    // Multiply U * S
    for (unsigned int i = 0; i < n; i++) {
        cublasSetStream(cublasHandle, stream);
        #ifdef USE_32BIT_TYPE
            check_cublas(cublasSscal(cublasHandle, n, S + i, U + i * n, 1), "cublasSscal");
        #else 
            check_cublas(cublasDscal(cublasHandle, n, S + i, U + i * n, 1), "cublasDscal");
        #endif
    }
    check_cuda(cudaStreamSynchronize(stream), "cublasDscal execute");

    // Multiply U by V (we multiply V by U^T to convert col to row major)
    check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    cublasSetStream(cublasHandle, stream);

    check_cublas(cublasDgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        n, n, n,
        &one, 
        V, n,
        U, n,
        &zero, 
        A, n), "cublasDgemm"
    );
    check_cuda(cudaStreamSynchronize(stream), "cublasDgemm execute");
}


void solveSvdGemm(const MAT_GPU &A, const MAT_GPU &B, MAT_GPU &X) {

  cudaStream_t v_stream;
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  check_cuda(cudaStreamCreate(&v_stream), "cudaStreamCreate");
  check_cublas(cublasCreate(&cublasHandle), "cublasCreate");
  check_cusolver(cusolverDnCreate(&cusolverHandle), "cusolverDnCreate");
  
  // Set stream
  cublasSetStream(cublasHandle, v_stream);
  cusolverDnSetStream(cusolverHandle, v_stream);

  // Allocate pseudoinverse array + work
  double *work;

  int work_int = 0;
  assert(A.n_rows == A.n_cols); // A should be square matrix

  int m = A.n_rows;
  int k = A.n_cols;
  int n = B.n_cols;

  check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
  gesvdjInfo_t gesvd_info;

  check_cusolver(cusolverDnCreateGesvdjInfo(&gesvd_info), "cusolverDnCreateGesvdjInfo");
  cusolverDnXgesvdjSetMaxSweeps(gesvd_info, 15); // As recommended by cuSOLVER docs

  check_cusolver(
    cusolverDnDgesvdj_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 1, m, m, A.vals, m, NULL, NULL, m, NULL, m, &work_int, gesvd_info), "cusolverDnDgesvdj_bufferSize");
  unsigned int work_length = 2 * m * m + m + work_int;
  check_cuda(cudaMalloc(&work, sizeof(double) * work_length), "cudaMalloc work");
  int * info;
  check_cuda(cudaMalloc(&info, sizeof(int)), "cudaMalloc info");
  // double * A
  // unsigned int n
  // double * work
  // unsigned int lwork
  // int * info
  // gesvdjInfo_t

  // Take pseudoinverse
  pseudoinverse_gpu(cusolverHandle, cublasHandle, v_stream, 
    A.vals, m, work, work_length, info, gesvd_info);

  // Multiply V^-1 by b
  check_cublas(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, 
    n, k, &one, A.vals, m, B.vals, k, &zero, X.vals, m), "cublasDgemm");

  // Clean up
  cublasDestroy(cublasHandle);
  cusolverDnDestroy(cusolverHandle);
  cusolverDnDestroyGesvdjInfo(gesvd_info);
  cudaStreamDestroy(v_stream);
  cudaFree(work);
  cudaFree(info);
}
