#include "cholesky_helpers.h"


__global__ void batch_copy_sqmatrix_to_rectmatrix(
    const double* sqmat, double* rectmat, 
    int sqmat_size, int rectmat_size) {
    extern __shared__ double shared_square[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sqmat_size) {
        shared_square[idx] = sqmat[idx];
    }
    __syncthreads();
    if (idx < rectmat_size) {
        rectmat[idx] = shared_square[idx%sqmat_size];
     }
}

__global__ void batch_copy_sqmatrix_to_rectmatrix_square_mat_oriented(
    const double* sqmat, double* rectmat, 
    int sqmat_size, int rectmat_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < rectmat_size) {
        rectmat[idx] = sqmat[idx%sqmat_size];
     }
}

__global__ void idxBasedCopyKernel(double * dst, double * src, unsigned int * ind, int n_rows, int n_cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_rows * n_cols && ind[idx] == 1) {
        dst[idx] = src[idx];
    }
}

__global__ void ucidxBasedCopyKernel(double * dst, double * src, unsigned char * ind, int n_rows, int n_cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_rows * n_cols) {
        if (ind[idx] == 1) {
            dst[idx] = src[idx];
        }
    } 
}

__global__ void apply_mask_to_batch_sqmatrices_kernel(double * rectmat, int sqmat_dim, int rectmat_size, unsigned int * masks) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rectmat_size) {
        int sqmat_size = sqmat_dim * sqmat_dim;
        int block_idx = idx / (sqmat_size);
        int idx_in_block = idx % sqmat_size;
        int row = idx_in_block % sqmat_dim;
        int col = idx_in_block / sqmat_dim;
        unsigned int * mask_vec = masks + block_idx * sqmat_dim; // apply proper offset

        if (row >= col) { // only the lower triangulr
            if (mask_vec[row] == 0) {
                if (row != col) {
                    rectmat[idx] = 0.0;
                } else {
                    rectmat[idx] = 1.0;
                }
            }
        } else {
            rectmat[idx] = 0.0;
        }
    }
}

__global__ void apply_uc_mask_to_batch_sqmatrices_kernel(double * rectmat, int sqmat_dim, int rectmat_size, unsigned char * masks) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rectmat_size) {
        int sqmat_size = sqmat_dim * sqmat_dim;
        int block_idx = idx / (sqmat_size);
        int idx_in_block = idx % sqmat_size;
        int row = idx_in_block % sqmat_dim;
        int col = idx_in_block / sqmat_dim;
        // unsigned char * mask_vec = masks + col_idx[block_idx] * sqmat_dim; // apply proper offset
        unsigned char * mask_vec = masks + block_idx * sqmat_dim; // apply proper offset

        if (row >= col) { // only the lower triangulr
            if (mask_vec[row] == 0 || mask_vec[col] == 0) {
                if (row != col) {
                    rectmat[idx] = 0.0;
                } else {
                    rectmat[idx] = 1.0;
                }
            }
        } else {
            rectmat[idx] = 0.0;
        }
    }
}
void create_batch_sqmatrices(double * mat_to_copy, double * batch_mats, size_t num_copys, size_t dims) {
    int rectmat_size = dims * dims * num_copys;
    int sqmat_size = dims * dims;


    int shared_mem_size = sizeof(double) * sqmat_size;
    // batch_copy_sqmatrix_to_rectmatrix<<<rectmat_size / BLOCK_SIZE + 1, BLOCK_SIZE, shared_mem_size, 0>>>(mat_to_copy, batch_mats, sqmat_size, rectmat_size);
    batch_copy_sqmatrix_to_rectmatrix_square_mat_oriented<<<rectmat_size / BLOCK_SIZE + 1, BLOCK_SIZE, 0, 0>>>(mat_to_copy, batch_mats, sqmat_size, rectmat_size);

}



/**
 * masks is size dims * num_cols
*/
void apply_mask_to_batch_sqmatrices(
    double * batch_sqmats, int dims, int num_cols, unsigned int * masks) {
    int rectmat_size = dims * dims * num_cols;
    apply_mask_to_batch_sqmatrices_kernel<<<rectmat_size / BLOCK_SIZE + 1, BLOCK_SIZE, 0, 0>>>(batch_sqmats, dims, rectmat_size, masks);
}

void apply_uc_mask_to_batch_sqmatrices(
    double * batch_sqmats, int dims, int num_cols, unsigned char * masks) {
    int rectmat_size = dims * dims * num_cols;
    apply_uc_mask_to_batch_sqmatrices_kernel<<<rectmat_size / BLOCK_SIZE + 1, BLOCK_SIZE, 0, 0>>>(batch_sqmats, dims, rectmat_size, masks);
}

int cusolverBatchedCholesky(double * matA_vals, double * matB_vals, uint block_size, uint num_blocks) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));

    int * infoArray = (int*)malloc(sizeof(int) * num_blocks); 

    // device pointers
    double **A_batch = (double**)malloc(sizeof(double*) * num_blocks);
    double **B_batch = (double**)malloc(sizeof(double*) * num_blocks);
    
    double **d_A_batch = nullptr;
    double **d_B_batch = nullptr;

    int *d_infoArray = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_infoArray, sizeof(int) * num_blocks));

    CHECK_CUDA(cudaMalloc((void**)&d_A_batch, sizeof(double*) * num_blocks));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_B_batch), sizeof(double*) * num_blocks));

    for (int i = 0; i < num_blocks; ++i) {
        A_batch[i] = &(matA_vals[i * block_size * block_size]);
        B_batch[i] = &(matB_vals[i * block_size]);
    }
    cudaMemcpy(d_A_batch, A_batch, sizeof(double*)*num_blocks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_batch, B_batch, sizeof(double*)*num_blocks, cudaMemcpyHostToDevice);

    /* step 3: Cholesky factorization */
    CHECK_CUSOLVER(
        cusolverDnDpotrfBatched(cusolverH, uplo, block_size, d_A_batch, block_size, d_infoArray, num_blocks));

    // assumes no problems
    CHECK_CUDA(cudaMemcpyAsync(infoArray, d_infoArray, sizeof(int) * num_blocks,
                               cudaMemcpyDeviceToHost, stream));
                            
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // for (int j = 0; j < num_blocks; j++) {
    //     std::printf("info[%d] = %d\n", j, infoArray[j]);
    // }
    CHECK_CUSOLVER(cusolverDnDpotrsBatched(cusolverH, uplo, block_size, 1, /* only support rhs = 1*/
                                           d_A_batch, block_size, d_B_batch, block_size, d_infoArray, num_blocks));

    CHECK_CUDA(cudaMemcpyAsync(infoArray, d_infoArray, sizeof(int), cudaMemcpyDeviceToHost,
                               stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // std::printf("after potrsBatched: infoArray[0] = %d\n", infoArray[0]);
    if (0 > infoArray[0]) {
        std::printf("%d-th parameter is wrong \n", -infoArray[0]);
        exit(1);
    }    
    return 0;
    // FREE EVERYTHING
}

void idxBasedCopy(double * dst, double * src, unsigned int * ind, int n_rows, int n_cols) {
    idxBasedCopyKernel<<<n_rows * n_cols / BLOCK_SIZE + 1, BLOCK_SIZE, 0, 0>>>(dst, src, ind, n_rows, n_cols);
}

void ucidxBasedCopy(double * dst, double * src, unsigned char * ind, int n_rows, int n_cols) {
    ucidxBasedCopyKernel<<<n_rows * n_cols / BLOCK_SIZE + 1, BLOCK_SIZE, 0, 0>>>(dst, src, ind, n_rows, n_cols);
}