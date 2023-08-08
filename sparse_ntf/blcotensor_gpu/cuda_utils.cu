#include "cuda_utils.h"

template <typename T>
T* make_device_copy(T* vector, _IType n, std::string name) {
    T* d_vector = nullptr;
    check_cuda(cudaMalloc(&d_vector, sizeof(T) * n), "cudaMalloc " + name);
    check_cuda(cudaMemcpy(d_vector, vector, sizeof(T) * n, cudaMemcpyHostToDevice), "cudaMemcpy " + name);
    return d_vector;
};
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
