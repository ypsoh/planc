#include "cuda_utils.h"


template _IType* make_device_copy(_IType* vector, _IType n, std::string name);
template _FType* make_device_copy(_FType* vector, _IType n, std::string name);
template _FType** make_device_copy(_FType** vector, _IType n, std::string name);
template _FType*** make_device_copy(_FType*** vector, _IType n, std::string name);
template unsigned int* make_device_copy(unsigned int* vector, _IType n, std::string name);
template unsigned long* make_device_copy(unsigned long* vector, _IType n, std::string name);
template int* make_device_copy(int* vector, _IType n, std::string name);

void check_cublas(cublasStatus_t status, std::string message) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error: " << cublasGetStatusString(status);
    std::cerr << ". " << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

void check_cuda(cudaError_t status, std::string message) {
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status);
    std::cerr << ". " << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

__global__ void transpose_mat(double * _mat_t, double * _mat, int m, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < m * n) {
    int orig_x = idx / m;
    int orig_y = idx % m;
    int dest_idx = orig_y * n + orig_x;
    _mat_t[dest_idx] = _mat[idx];
  }
}

// Hadamard update, i.e. x <-- x .* y
__global__ void hadamard_kernel(_FType* x, _FType* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) x[index] *= y[index];
}
void vector_hadamard(_FType* x, _FType* y, int n) {
    hadamard_kernel <<<n / BLOCK_SIZE + 1, BLOCK_SIZE, 0>>>(x, y, n);
    check_cuda(cudaGetLastError(), "hadamard_kernel launch");
}

// Fills every element of vector x with the same value
__global__ void value_fill_kernel(_FType* x, _IType n, _FType val) {
    _IType index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) x[index] = val;
}
void value_fill(_FType* x, _IType n, _FType val) {
    value_fill_kernel <<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(x, n, val);
    check_cuda(cudaGetLastError(), "value_fill_kernel launch");
}

__global__ void normalizeColumn(double* matrix, double* lambda, int numRows, int numCols) {
    // chooses column
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank < numCols) {
        double * col = matrix + rank * numRows;
        // Calculate the Frobenius norm of the column
        double norm = 0.0;
        for (int i = 0; i < numRows; ++i) {
            norm += col[i] * col[i];
        }
        norm = sqrtf(norm);

        for (int i = 0; i < numRows; ++i) {
            col[i] /= norm;
        }
        // Normalize the element in the column
        // matrix[index] /= norm;
        lambda[rank] = norm;
    }
}

void normalize_fm(MAT_GPU * fm, double * lambda) {
  int rank = fm->n_cols;
  int dim = fm->n_rows;

  int blockSize = 256;
  int gridSize = (rank + blockSize - 1) / blockSize;

  normalizeColumn<<<gridSize, blockSize>>>(fm->vals, lambda, dim, rank);
  check_cuda(cudaDeviceSynchronize(), "sync after normaliztation");
}

void mat_mat_mul(_FType* a, _FType* b, _FType* c, int m, int n, int k, double alpha, double beta) {
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");
  check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
  m, n, k, &alpha, a, m, b, k, &beta, c, m), "mat_mat_mul");
  cublasDestroy(handle);
}

void mat_vec_mul(_FType* a, _FType* b, _FType* c, int m, int n, double alpha, double beta) {
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");
  check_cublas(cublasDgemv(handle, CUBLAS_OP_N, 
  m, n, &alpha, a, m, b, 1, &beta, c, 1), "mat_vec_mul");
  cublasDestroy(handle);
}

__global__ void vec_add_sub(double* v1, const double* v2, int n, bool add) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (add) {
      v1[idx] += v2[idx]; // Element-wise addition
    } else {
      v1[idx] -= v2[idx]; // Element-wise subtraction
    }
  }
}

__global__ void fix_numerical_error(double* v, int n, const double th, const double repl) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    v[idx] = (v[idx] < th) ? repl : v[idx];
  }
}