#include "bpp.h"



/**
 * pset = X > 0
 * non_opt_set = Y < 0 && X <= 0
 * infea_set = X < 0 && X > 0
*/
__global__ void init_passive_non_opt_infea_sets_kernel(unsigned char * pset_vals, 
    unsigned char * non_opt_set_vals, unsigned char * infeas_set_vals, double * d_X, double * d_Y, unsigned int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        double x_val = d_X[idx];
        double y_val = d_Y[idx];

        unsigned char is_x_strictly_pos = x_val > 0.0; // Extract the sign bit
        unsigned char is_x_zero = x_val == 0.0;
        unsigned char is_y_strictly_pos = y_val > 0.0;
        unsigned char is_y_zero = y_val == 0.0;
        
        pset_vals[idx] = is_x_strictly_pos;
        non_opt_set_vals[idx] = (!is_y_strictly_pos & !is_y_zero) & !is_x_strictly_pos;
        infeas_set_vals[idx] = (!is_x_strictly_pos & !is_x_zero) & is_x_strictly_pos;
    }  
}

__global__ void init_alpha_beta_vec_kernel(unsigned char * num_trial, unsigned int * num_infeas_val, int nrhs, int rank, unsigned char default_num_trials) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrhs) {
        num_trial[idx] = default_num_trials;
        num_infeas_val[idx] = rank + 1;
    }
}

__global__ void compute_two_ucmat_column_sum_kernel(unsigned char * ucmat1, unsigned char * ucmat2, 
    unsigned int * col_sum_vec, int rank, int nrhs) {
    int colidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (colidx < nrhs) {
        unsigned int sum = 0;
        for (int r = 0; r < rank; ++r) {
            sum += ucmat1[colidx * rank + r] + ucmat2[colidx * rank + r];
        }
        col_sum_vec[colidx] = sum;
    }
}

__global__ void compute_uivec_sum_kernel(unsigned int * col_vec, unsigned int * sum_col_vec, int vec_size) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < vec_size && col_vec[i] > 0) ? 1 : 0;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Write the result for this block to global memory
    if (tid == 0) atomicAdd(sum_col_vec, sdata[0]);    
}

__global__ void update_case_vec_kernel(unsigned int * non_opt_cols, unsigned char * case_ind_vec, unsigned char * num_trial_val, unsigned int * num_infeas_val, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        if ((non_opt_cols[idx] > 0) && (non_opt_cols[idx] < num_infeas_val[idx])) {
            case_ind_vec[idx] = 1;
        } 
        else if((non_opt_cols[idx] >= num_infeas_val[idx]) && num_trial_val[idx] >= 1) {
            case_ind_vec[idx] = 2;
        } 
        else if ((non_opt_cols[idx] > 0)){
            case_ind_vec[idx] = 3;
        }
        else case_ind_vec[idx] = 0;
    }
}

__global__ void update_partition_kernel_p1(unsigned int * non_opt_cols, unsigned char * case_ind_vec, unsigned char * num_trial_val, unsigned int * num_infeas_val, unsigned int nrhs) {
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (col_idx < nrhs) {
        // update col_idx
        unsigned char case_ind = case_ind_vec[col_idx];
        if (case_ind == 1) {
            num_trial_val[col_idx] = 3; // replenish num trials
            num_infeas_val[col_idx] = non_opt_cols[col_idx]; // num of infeasible values is updated
        }
        else if (case_ind == 2) {
            num_trial_val[col_idx]--; // reduce num_trial_val by one
            num_infeas_val[col_idx] = non_opt_cols[col_idx]; // num of infeasible values is updated
        }
        else if (case_ind == 3) {
            printf("not considering case 3 for now...\n");
        }
    }
}

__global__ void update_partition_kernel_p2(unsigned char * pset_vals, unsigned char * non_opt_vals, unsigned char * infeas_set_vals, unsigned char * case_ind_vec, unsigned int rank, unsigned int nrhs) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrhs * rank) {
        unsigned int col_idx = idx / rank;
        unsigned char case_ind = case_ind_vec[col_idx];

        if (case_ind == 1 || case_ind == 2) {
            if (non_opt_vals[idx] == 1) pset_vals[idx] = 1;
            if (infeas_set_vals[idx] == 1) pset_vals[idx] = 0;
        }
    }
}
void init_passive_non_opt_infea_sets(unsigned char * pset_vals, 
    unsigned char * non_opt_set_vals, unsigned char * infeas_set_vals, double * d_X, double * d_Y, unsigned int size) {
    init_passive_non_opt_infea_sets_kernel<<<size/BLOCK_SIZE+1, BLOCK_SIZE, 0, 0>>>(pset_vals, non_opt_set_vals, infeas_set_vals, d_X, d_Y, size);
}

void init_alpha_beta_vec(unsigned char * num_trial, unsigned int * num_infeas_val, int nrhs, int rank, unsigned char default_num_trials) {
    init_alpha_beta_vec_kernel<<<nrhs/BLOCK_SIZE+1, BLOCK_SIZE, 0, 0>>>(num_trial, num_infeas_val, nrhs, rank, default_num_trials);
}

void compute_two_ucmat_column_sum(unsigned char * ucmat1, unsigned char * ucmat2, 
    unsigned int * col_sum_vec, int rank, int nrhs) {
        compute_two_ucmat_column_sum_kernel<<<nrhs/BLOCK_SIZE+1, BLOCK_SIZE, 0, 0>>>(ucmat1, ucmat2, col_sum_vec, rank, nrhs);
}

void compute_uivec_sum(unsigned int * col_vec, unsigned int * sum_col_vec, int vec_size) {
    compute_uivec_sum_kernel<<<vec_size/BLOCK_SIZE+1, BLOCK_SIZE, sizeof(unsigned int) * BLOCK_SIZE, 0>>>(col_vec, sum_col_vec, vec_size);
}

void update_case_vec(unsigned int * non_opt_cols, unsigned char * case_ind_vec, unsigned char * num_trial_val, unsigned int * num_infeas_val, int size) {
    update_case_vec_kernel<<<size/BLOCK_SIZE+1, BLOCK_SIZE, 0, 0>>>(non_opt_cols, case_ind_vec, num_trial_val, num_infeas_val, size);
}

// Update on a column basis
void update_partition_p1(unsigned int * non_opt_cols, unsigned char * case_ind_vec, unsigned char * num_trial_val, unsigned int * num_infeas_val, unsigned int nrhs) {
    update_partition_kernel_p1<<<nrhs/BLOCK_SIZE+1, BLOCK_SIZE, 0, 0>>>(non_opt_cols, case_ind_vec, num_trial_val, num_infeas_val, nrhs);
}

// Update on a element wise basis -- update passive set
void update_partition_p2(unsigned char * pset_vals, unsigned char * non_opt_vals, unsigned char * infeas_set_vals, unsigned char * case_ind_vec, unsigned int rank, unsigned int nrhs) {
    update_partition_kernel_p2<<<nrhs * rank/BLOCK_SIZE+1, BLOCK_SIZE, 0, 0>>>(pset_vals, non_opt_vals, infeas_set_vals, case_ind_vec, rank, nrhs);
}
