#include <iostream>
#include "common/utils.h"
#include "common/ncpfactors.hpp"
#include "common/ntf_utils.hpp"
#include "common/parsecommandline.hpp"
#include "common/sparse_tensor.hpp"
#include "common/alto_tensor.hpp"
#include "common/blco_tensor.hpp"
#include "nnls/bppnnls.hpp"
#include "cuda_utils.h"
#include "blco.h"
#include "vector_ops.h"
#include "matrix_ops.h"
#include "aoadmm.h"
#include "cublas_operations.h"
#include "vector_ops.h"
#include "bpp.h"
#include "cholesky_helpers.h"
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <unordered_map>

// BPPRecursivePartitionSolve
#include <bitset>


#if ALTO_MASK_LENGTH == 64
  typedef unsigned long long LIType;
#elif ALTO_MASK_LENGTH == 128
  typedef unsigned __int128 LIType;
#else
  #pragma message("Using default 128-bit.")
  typedef unsigned __int128 LIType;
  // typedef unsigned long long LIType;

#endif

#define RANK 4
#define NROW 10
#define UUMAT arma::Mat<uint32_t>
#define UUVEC arma::Row<uint32_t>
#define BMAT arma::Mat<unsigned char>
#define BVEC arma::Row<unsigned char>
#define DEBUG 0

#define FP64_PEAK_FLOPS 9.7e12
#define FP64_TC_PEAK_FLOPS 19.5e12

struct cudaEventPair {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventPair() {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
  }
  ~cudaEventPair() {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }
  void tic() {
      cudaEventRecord(start);
  }
  void toc() {
      cudaEventRecord(end);
  }

  void sync() {
      cudaEventSynchronize(end);
  }

  float getDuration() {
      float time;
      cudaEventElapsedTime(&time, start, end);
      return time;
  }
};

int profile_cpu_bpp_nnls(int rank, int nrhs) {
  INFO << "Profiling CPU BPP based NNLS solve" << "\n";

  MAT fm = arma::randn<MAT>(nrhs, rank);
  MAT gram = fm.t() * fm;
  MAT mttkrp_t = arma::randn<MAT>(rank, nrhs); // mttkrp.t()
  
  int _block_size = nrhs;
  UINT numChunks = nrhs / _block_size;
  if (numChunks * _block_size < nrhs) numChunks++;

  // #pragma omp parallel for schedule(dynamic) if(numChunks > 1)
  // #pragma omp parallel for schedule(dynamic) reduction(+:it) if(numChunks > 1)
  for (UINT i = 0; i < numChunks; i++) {
    UINT spanStart = i * _block_size;
    UINT spanEnd = (i + 1) * _block_size - 1;
    if (spanEnd > nrhs - 1) {
      spanEnd = nrhs - 1;
    }

    BPPNNLS<MAT, VEC> subProblem(gram,
                      (MAT)mttkrp_t.cols(spanStart, spanEnd),
                      true);
    int num_iters = subProblem.solveNNLS();
    INFO << num_iters << std::endl;
  }
  
  // subProblem.solveNNLS();
  // INFO << subProblem.getSolutionMatrix() << std::endl;
  return 0;
}


// Mirroring the BPPNNLS implementaiton
int main(int argc, char* argv[]) {
  std::string is_cpu_gpu = argv[1];
  int rank = std::stoi(argv[2]);
  int nrhs = std::stoi(argv[3]);

  INFO << "Investigating BPP..." << std::endl;
  arma::arma_rng::set_seed(13);

  MAT fm = arma::randn<MAT>(rank, nrhs);
  MAT gram = fm * fm.t();
  MAT mttkrp_t = arma::randn<MAT>(rank, nrhs); // mttkrp.t()

  if (is_cpu_gpu == "cpu") {
    INFO << "==== CPU part ====" << std::endl;
    
    BPPNNLS<MAT, VEC> subProblem(gram, (MAT)mttkrp_t, true);
    subProblem.solveNNLS();
    // INFO << subProblem.getSolutionMatrix() << std::endl;
  }

  if (is_cpu_gpu == "gpu") {

  INFO << "==== GPU part ====" << std::endl;

  cudaError_t cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
    return 1;
  }

  size_t full_mat_size = nrhs * rank;
  size_t gram_mat_size = rank * rank;

  // Full precision vectors
  double * d_fm_vals, * d_mttkrp_vals, * d_gram_vals;
  double * d_Y, * d_X;
  
  // bool mats -- bitops needs further consideration, use 1 byte for now
  unsigned char * pset_vals;
  unsigned char * non_opt_set_vals;
  unsigned char * infeas_set_vals;

  // Case flag for each column
  unsigned char * case_ind_vec;

  // size nrhs vectors
  unsigned char * num_trial_val; // number of trials before using single variable partition
  unsigned int * num_infeas_val; // number of infeasible variables per column
  unsigned int * non_opt_cols;

  // For debugging
  BMAT _bmat = arma::zeros<BMAT>(rank, nrhs);
  BVEC _bvec = arma::zeros<BVEC>(nrhs);
  MAT _mat = arma::zeros<MAT>(rank, nrhs);
  arma::Row<unsigned int> _uvec = arma::zeros<arma::Row<unsigned int>>(nrhs);
  // arma::Mat<bool> _bmat(rank, nrhs);
  
  // malloc/copy over to gpu
  cudaMallocAsync((void**)&d_fm_vals, sizeof(double) * full_mat_size, 0);
  cudaMallocAsync((void**)&d_mttkrp_vals, sizeof(double) * full_mat_size, 0);
  cudaMallocAsync((void**)&d_gram_vals, sizeof(double) * gram_mat_size, 0);

  cudaMallocAsync((void**)&d_Y, sizeof(double) * full_mat_size, 0);
  cudaMallocAsync((void**)&d_X, sizeof(double) * full_mat_size, 0);

  cudaMallocAsync((void**)&pset_vals, sizeof(unsigned char) * full_mat_size, 0);
  cudaMallocAsync((void**)&non_opt_set_vals, sizeof(unsigned char) * full_mat_size, 0);
  cudaMallocAsync((void**)&infeas_set_vals, sizeof(unsigned char) * full_mat_size, 0);
  
  // unsigned char vector that indicates the "case (1, 2, 3)" for each column (nrhs)
  cudaMallocAsync((void**)&case_ind_vec, sizeof(unsigned char) * nrhs, 0);
  
  cudaMallocAsync((void**)&num_trial_val, sizeof(unsigned char) * nrhs, 0);
  cudaMallocAsync((void**)&num_infeas_val, sizeof(unsigned int) * nrhs, 0);
  cudaMallocAsync((void**)&non_opt_cols, sizeof(unsigned int) * nrhs, 0);


  cudaMemcpyAsync(d_fm_vals, fm.memptr(), sizeof(double) * full_mat_size, cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(d_mttkrp_vals, mttkrp_t.memptr(), sizeof(double) * full_mat_size, cudaMemcpyHostToDevice, 0); 
  cudaMemcpyAsync(d_gram_vals, gram.memptr(), sizeof(double) * gram_mat_size, cudaMemcpyHostToDevice, 0); 

  cudaMemsetAsync(d_X, 0, sizeof(double) * full_mat_size, 0);
  // cudaMemcpyAsync(d_Y, fm.memptr(), sizeof(double) * full_mat_size, cudaMemcpyHostToDevice, 0);

  cudaEventPair ev[5];
  double ev0 = 0.0; // mat ops
  double ev1 = 0.0; // check feas
  double ev2 = 0.0; // upd part
  double ev3 = 0.0; // prepare chol
  double ev4 = 0.0; // solve chol

  ev[0].tic();
  cublasHandle_t handle_;
  CHECK_CUBLAS(cublasCreate(&handle_));
  // attach stream if necessary

  double alpha_ = 1.0;
  double beta_ = 0.0;

  CHECK_CUBLAS(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, rank, nrhs, rank,
  &alpha_, d_gram_vals, rank, d_X, rank, &beta_, d_Y, rank));

  alpha_ = 1.0;
  beta_ = -1.0;

  CHECK_CUBLAS(cublasDgeam(handle_, CUBLAS_OP_N, CUBLAS_OP_N, rank, nrhs, 
  &alpha_, d_Y, rank, &beta_, d_mttkrp_vals, rank, 
  d_Y, rank));

  cublasDestroy(handle_);
  ev[0].toc();
  ev[0].sync();
  ev0 += ev[0].getDuration();
  // PassiveSet = X > 0
  // NonOptSet & InfeasSet
  // UMAT_GPU LO = * Y < 0;
  // UMAT_GPU RO = PassiveSet == 0;

  // UMAT_GPU negX = *X < 0;

  // UMAT_GPU NonOptSet = LO % RO;
  // UMAT_GPU InfeaSet = negX % PassiveSet;

  ev[1].tic();
  init_passive_non_opt_infea_sets(pset_vals, non_opt_set_vals, infeas_set_vals, d_X, d_Y, full_mat_size);
  init_alpha_beta_vec(num_trial_val, num_infeas_val, nrhs, rank, 3);

  // INFO << mttkrp_t << "\n";

  // cudaMemcpy(_bmat.memptr(), pset_vals, sizeof(unsigned char) * full_mat_size, cudaMemcpyDeviceToHost);
  // INFO << _bmat << "\n";
  // cudaMemcpy(_bmat.memptr(), non_opt_set_vals, sizeof(unsigned char) * full_mat_size, cudaMemcpyDeviceToHost);
  // INFO << _bmat << "\n";
  // cudaMemcpy(_bmat.memptr(), infeas_set_vals, sizeof(unsigned char) * full_mat_size, cudaMemcpyDeviceToHost);
  // INFO << _bmat << "\n";
  
  compute_two_ucmat_column_sum(non_opt_set_vals, infeas_set_vals, non_opt_cols, rank, nrhs);
  cudaMemcpy(_uvec.memptr(), non_opt_cols, sizeof(unsigned int) * nrhs, cudaMemcpyDeviceToHost);
  UVEC col_idx_to_solve = arma::find(_uvec);
  // UUVEC col_idx_to_solve(_col_idx_to_solve);
  // 
  unsigned int num_cols_to_solve = col_idx_to_solve.size();
  // unsigned int _MAX_ITERATIONS = nrhs * 5;
  unsigned int _MAX_ITERATIONS = 1;
  bool _success = true;

  ev[1].toc();
  ev[1].sync();
  ev1 += ev[1].getDuration();

  unsigned int _iter = 0;

  while (num_cols_to_solve > 0) {
    _iter++;
    if (_MAX_ITERATIONS > 0 && _iter > _MAX_ITERATIONS) {
      _success = false;
      printf("bpp failed\n");
      break;
    }
    ev[2].tic();
    // Update partitions
    // case_ind_vec will have entries of 1, 2, 3 based on the following conditions
    // case 1: NotOptCols % (NotGood < Ninf) -- The number of not good variables have decreased
    // case 2: NotOptCols % (NotGood >= Ninf) % (P >= 1) -- The number of not good variables have increased
    // case 3: NotOptCols % (Not Case 1 nor Case 2)
    update_case_vec(non_opt_cols, case_ind_vec, num_trial_val, num_infeas_val, nrhs);

    // cudaMemcpy(_bvec.memptr(), case_ind_vec, sizeof(unsigned char) * nrhs, cudaMemcpyDeviceToHost);
    // INFO << _bvec << "\n";

    // Update non_opt_set and infea_set_vals -- partition
    update_partition_p1(non_opt_cols, case_ind_vec, num_trial_val, num_infeas_val, nrhs);
    update_partition_p2(pset_vals, non_opt_set_vals, infeas_set_vals, case_ind_vec, rank, nrhs);

    ev[2].toc();
    ev[2].sync();
    ev2 += ev[2].getDuration();
    // cudaDeviceSynchronize();

    // passive set is being properly updated...
    // cudaMemcpy(_bmat.memptr(), pset_vals, sizeof(unsigned char) * full_mat_size, cudaMemcpyDeviceToHost);
    // INFO << _bmat << "\n";

    // Solve AX=B where pset_vals is the PassiveSet (masks)
    double * batched_L;
    cudaMallocAsync((void**)&batched_L, sizeof(double) * num_cols_to_solve * rank * rank, 0);
    ev[3].tic();
    // INFO << num_cols_to_solve << "\n";
    // MAT _batched_L = arma::zeros<MAT>(rank, num_cols_to_solve * rank);
    create_batch_sqmatrices(d_gram_vals, batched_L, num_cols_to_solve, rank);
    cudaDeviceSynchronize();
    // cudaMemcpy(_batched_L.memptr(), batched_L, sizeof(double) * rank * rank * num_cols_to_solve, cudaMemcpyDeviceToHost);
    // INFO << _batched_L << "\n";

    apply_uc_mask_to_batch_sqmatrices(batched_L, rank, num_cols_to_solve, pset_vals);
    cudaDeviceSynchronize();

    // cudaMemcpy(_batched_L.memptr(), batched_L, sizeof(double) * rank * rank * num_cols_to_solve, cudaMemcpyDeviceToHost);
    // INFO << _batched_L << "\n";

    // copy o_mttkrp_gpu to X
    ucidxBasedCopy(d_X, d_mttkrp_vals, pset_vals, rank, num_cols_to_solve);
    ev[3].toc();
    ev[3].sync();
    ev3 += ev[3].getDuration();

    // cudaMemcpy(_mat.memptr(), d_X, sizeof(double) * rank * num_cols_to_solve, cudaMemcpyDeviceToHost);
    // INFO << _mat << "\n";

    ev[4].tic();
    cusolverBatchedCholesky(batched_L, d_X, rank, num_cols_to_solve);
    ev[4].toc();
    ev[4].sync();
    ev4 += ev[4].getDuration();
    // cudaDeviceSynchronize();

    // cudaMemcpy(_mat.memptr(), d_X, sizeof(double) * rank * nrhs, cudaMemcpyDeviceToHost);
    // INFO << _mat << "\n";

    // Update Y
    // Y = gram * X
    // - mttkrp

    // fix numerical error for X
    // fix numerical error for y0

    // update non_opt_set_vals and infeas_set_vals

    // compute non_opt_cols given updated non_opt_set_vals and infeas_set_vals

    ev[1].tic();
    compute_two_ucmat_column_sum(non_opt_set_vals, infeas_set_vals, non_opt_cols, rank, nrhs);
    cudaMemcpy(_uvec.memptr(), non_opt_cols, sizeof(unsigned int) * nrhs, cudaMemcpyDeviceToHost);
    col_idx_to_solve = arma::find(_uvec);
    // col_idx_to_solve(_col_idx_to_solve);
    num_cols_to_solve = col_idx_to_solve.size();
    // INFO << _uvec << "\n";
    // INFO << num_cols_to_solve << "\n";
    ev[1].toc();
    ev[1].sync();
    ev1 += ev[1].getDuration();
  }

  // ev0 -- mat ops
  // ev1 -- feas check
  // ev2 -- preparing batched cholesky
  // ev3 -- batch solve
  printf("matops\t%f\nfeas_check\t%f\nupdate_part\t%f\nprep_batch\t%f\nbatch_solve\t%f\n", 2*ev0, ev1, ev2, ev3, ev4);

  cudaFree(d_fm_vals);
  cudaFree(d_mttkrp_vals);
  cudaFree(d_gram_vals);
  cudaFree(d_Y);
  cudaFree(d_X);


  cudaFree(pset_vals);
  cudaFree(non_opt_set_vals);
  cudaFree(infeas_set_vals);

  cudaFree(case_ind_vec);

  cudaFree(num_trial_val);
  cudaFree(num_infeas_val);
  cudaFree(non_opt_cols);

  exit(0);
  // 

  MAT_GPU * fm_gpu = send_mat_to_gpu(&fm);
  MAT_GPU * mttkrp_t_gpu = send_mat_to_gpu(&mttkrp_t);
  MAT_GPU * gram_gpu = send_mat_to_gpu(&gram);

  MAT_GPU * sub_mttkrp = mttkrp_t_gpu;
  MAT_GPU * Y = init_mat_gpu(sub_mttkrp->n_rows, sub_mttkrp->n_cols);
  MAT_GPU * X = init_mat_gpu(sub_mttkrp->n_rows, sub_mttkrp->n_cols); // solution matrix
  cudaMemset(X->vals, 0.0, sizeof(double) * X->n_rows * X->n_cols);

  nrhs = Y->n_cols; // # of rhs
  unsigned int k = nrhs; // rank
  unsigned int n = Y->n_rows; 
  // 
  mat_mat_mul(gram_gpu->vals, X->vals, Y->vals, gram_gpu->n_rows, gram_gpu->n_cols, X->n_cols, 1.0, 0.0);

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");

  double alpha = 1.0;
  double beta = -1.0;

  check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Y->n_rows, Y->n_cols, 
  &alpha, Y->vals, Y->n_rows, &beta, sub_mttkrp->vals, sub_mttkrp->n_rows, 
  Y->vals, Y->n_rows), "Y <- 1.0 * Y - 1.0 * sub_mttkrp");

  cublasDestroy(handle);

  // For debugging
  MAT debug_mat = arma::zeros<MAT>(X->n_rows, X->n_cols);
  MAT debug_gram = arma::zeros<MAT>(rank, rank);
  MAT debug_vector = arma::zeros<MAT>(rank, 1);
  UUMAT debug_umat = arma::zeros<UUMAT>(X->n_rows, X->n_cols);
  UUVEC debug_uvec = arma::zeros<UUVEC>(X->n_cols);
  
#if DEBUG==1
  send_mat_to_host(Y, &debug_mat);
  INFO << "Y\n" << debug_mat << std::endl; 

  send_mat_to_host(X, &debug_mat);
  INFO << "X\n" << debug_mat << std::endl; 

  send_mat_to_host(sub_mttkrp, &debug_mat);
  INFO << "sub_mttkrp\n" << debug_mat << std::endl; 
#endif
  UMAT_GPU PassiveSet = *X > 0;
  UMAT_GPU LO = * Y < 0;
  UMAT_GPU RO = PassiveSet == 0;

  UMAT_GPU negX = *X < 0;

  UMAT_GPU NonOptSet = LO % RO;
  UMAT_GPU InfeaSet = negX % PassiveSet;

  UVEC_GPU P = UVEC_GPU(k);
  UVEC_GPU Ninf = UVEC_GPU(k);
  unsigned int pbar = 3;
  P.fill(pbar);
  Ninf.fill(n+1); // n+1

#if DEBUG==1
  send_umat_to_host(&NonOptSet, &debug_umat);
  INFO << "NonOptSet\n" << debug_umat << "\n";
  send_umat_to_host(&InfeaSet, &debug_umat);
  INFO << "InfeaSet\n" << debug_umat << "\n";
#endif
  UVEC_GPU SumNonOptSet = NonOptSet.sum();
  UVEC_GPU SumInfeaSet = InfeaSet.sum();

  UVEC_GPU NotGood = SumNonOptSet + SumInfeaSet; // need to add InfeaSet.sum();
  cudaMemcpy(debug_uvec.memptr(), NotGood.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
  INFO << "NotGood\n" << debug_uvec << "\n";

  UVEC_GPU NotOptCols = NotGood > 0;
  cudaMemcpy(debug_uvec.memptr(), NotOptCols.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
  INFO << "NotOptCols\n" << debug_uvec << "\n";

  unsigned int numNonOptCols = NotOptCols.sum();
  INFO << "numNonOptCols: " << NotOptCols.sum() << std::endl;

  unsigned int MAX_ITERATIONS = nrhs * 5;
  // unsigned int MAX_ITERATIONS = 3;
  bool success = true;

  UVEC_GPU Cols1 = UVEC_GPU(k);
  UVEC_GPU Cols2 = UVEC_GPU(k);
  UVEC_GPU Cols3 = UVEC_GPU(k);

  UMAT_GPU PSetBits = NonOptSet;
  UMAT_GPU POffBits = InfeaSet;
  // UMAT_GPU NotOptMask = UMAT_GPU(NonOptSet.n_rows, NonOptSet.n_cols);
  // UMAT_GPU   

  unsigned int iter = 0;
  while(numNonOptCols > 0) {
    iter++;
    if ((MAX_ITERATIONS > 0) && (iter > MAX_ITERATIONS)) {
      success = false;
      break;
    }
    // cudaMemcpy(debug_uvec.memptr(), NotOptCols.vals, sizeof(uint32_t) * NotOptCols.size, cudaMemcpyDeviceToHost);
    // INFO << "NotOptCols\n" << debug_uvec << "\n";

    Cols1 = NotOptCols % (NotGood < Ninf);
    cudaMemcpy(debug_uvec.memptr(), Cols1.vals, sizeof(uint32_t) * Cols1.size, cudaMemcpyDeviceToHost);
    INFO << "Cols1\n" << debug_uvec << "\n";

    Cols2 = NotOptCols % (NotGood >= Ninf) % (P >= 1);
    cudaMemcpy(debug_uvec.memptr(), Cols2.vals, sizeof(uint32_t) * Cols2.size, cudaMemcpyDeviceToHost);
    INFO << "Cols2\n" << debug_uvec << "\n";

#if DEBUG==0
    send_umat_to_host(&NonOptSet, &debug_umat);
    INFO << "NonOptSet\n" << debug_umat << "\n";

    send_umat_to_host(&InfeaSet, &debug_umat);
    INFO << "InfeaSet\n" << debug_umat << "\n";
#endif
    // P.fill(99);
    // cudaMemcpy(debug_uvec.memptr(), P.vals, sizeof(uint32_t) * P.size, cudaMemcpyDeviceToHost);
    // INFO << "P\n" << debug_uvec << "\n";

    // Compute Cols3 that aren't included by Cols1 and Cols2
    // Cols3 = NotOptCols % (Cols1 == 0);
    // Cols3 = Cols3 % (Cols2 == 0);

    // cudaMemcpy(debug_uvec.memptr(), Cols3.vals, sizeof(uint32_t) * Cols3.size, cudaMemcpyDeviceToHost);
    // INFO << "Cols3\n" << debug_uvec << "\n";
    // P.fill(33);
#if DEBUG==1
    cudaMemcpy(debug_uvec.memptr(), P.vals, sizeof(uint32_t) * P.size, cudaMemcpyDeviceToHost);
    INFO << "P\n" << debug_uvec << "\n";
#endif
    
    if (!Cols1.empty()) {
      INFO << "Cols1 not empty" << "\n";
      // Replenish P values to pbar (3)
      // Replenish Ninf values to NotGood

      P.idx_based_fill(Cols1, pbar);
      Ninf.idx_based_copy(Cols1, NotGood);

#if DEBUG==1
    cudaMemcpy(debug_uvec.memptr(), Ninf.vals, sizeof(uint32_t) * Ninf.size, cudaMemcpyDeviceToHost);
    INFO << "Ninf\n" << debug_uvec << "\n";
#endif

      PSetBits = NonOptSet;
      PSetBits.rowwise_mult(Cols1); // UMAT_GPU.rowsize_mult(UVEC_GPU&)
#if DEBUG==0
    send_umat_to_host(&PSetBits, &debug_umat);
    INFO << "PSetBits\n" << debug_umat << "\n";
#endif
      PassiveSet.idx_based_fill(PSetBits, 1u);

      POffBits = InfeaSet;
      POffBits.rowwise_mult(Cols1);
      PassiveSet.idx_based_fill(POffBits, 0u);
    }

    if (!Cols2.empty()) {
      INFO << "Cols2 not empty" << "\n";

      P.idx_based_sub(Cols2, 1);

#if DEBUG==1
    cudaMemcpy(debug_uvec.memptr(), Ninf.vals, sizeof(uint32_t) * Ninf.size, cudaMemcpyDeviceToHost);
    INFO << "Ninf\n" << debug_uvec << "\n";
#endif
#if DEBUG==0
    send_umat_to_host(&NonOptSet, &debug_umat);
    INFO << "NonOptSet-Col2\n" << debug_umat << "\n";
#endif
      PSetBits = NonOptSet;

      PSetBits.rowwise_mult(Cols2); // UMAT_GPU.rowsize_mult(UVEC_GPU&)
#if DEBUG==0
    send_umat_to_host(&PSetBits, &debug_umat);
    INFO << "PSetBits\n" << debug_umat << "\n";
#endif
      PassiveSet.idx_based_fill(PSetBits, 1u);

      POffBits = InfeaSet;
      POffBits.rowwise_mult(Cols2);
      PassiveSet.idx_based_fill(POffBits, 0u);
    }

    // This case is used as a fall back
    // hoping for now this fallback cases does not occur
    // add on implementation afterwards
    if (!Cols3.empty()) {
      INFO << "Cols3 not empty" << "\n";
    }

    send_umat_to_host(&PassiveSet, &debug_umat);
    INFO << "PassiveSet\n" << debug_umat << "\n";

#if DEBUG==1
    send_mat_to_host(gram_gpu, &debug_gram);
    INFO << "gram_gpu\n" << debug_gram << std::endl; 
#endif
    // It should only try to solve Ax=b for columns that have nonOptCols
    cudaMemcpy(debug_uvec.memptr(), NotOptCols.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
    for (int col_idx = 0; col_idx < nrhs; ++col_idx) {
      if (debug_uvec[col_idx] == 1) {
        MAT_GPU MASKED_A = gram_gpu->apply_mask(PassiveSet.vals + col_idx * RANK);
        MAT_GPU b = sub_mttkrp->col(col_idx);
        MAT_GPU MASKED_b = b.apply_mask(PassiveSet.vals + col_idx * RANK);

#if DEBUG==1 // for debugging purposes
        if (col_idx == 1) {
          send_mat_to_host(&MASKED_A, &debug_gram);
          INFO << "MASKED_A\n" << debug_gram << std::endl; 

          send_mat_to_host(&MASKED_b, &debug_vector);
          INFO << "MASKED_b\n" << debug_vector << std::endl; 
        }
#endif

        MAT_GPU MASKED_X(MASKED_b.n_rows, MASKED_b.n_cols);
        solveSvdGemm(MASKED_A, MASKED_b, MASKED_X);

#if DEBUG==1 // for debugging purposes
        if (col_idx == 1) {
          send_mat_to_host(&MASKED_X, &debug_vector);
          INFO << "MASKED_X\n" << debug_vector << std::endl; 
        }
#endif

        X->masked_col_update(
        // Update X one column vector at a time
          col_idx,
          PassiveSet.vals + col_idx * RANK, 
          MASKED_X.vals);

        // Update Y here
      }
    } // Update this->X complete
    send_mat_to_host(X, &debug_mat);
    INFO << "X\n" << debug_mat << std::endl; 

    // Verified
    // send_mat_to_host(X, &debug_mat);
    // INFO << "after solve: X\n" << debug_mat << std::endl; 
    
    /* Y = gram * X */
    mat_mat_mul(gram_gpu->vals, X->vals, Y->vals, gram_gpu->n_rows, gram_gpu->n_cols, X->n_cols, 1.0, 0.0);

#if DEBUG==1
    // send_mat_to_host(X, &debug_mat);
    // INFO << "after numerical error fix: X\n" << debug_mat << std::endl; 
    // // Verified
    send_mat_to_host(Y, &debug_mat);
    INFO << "gram * X\n" << debug_mat << std::endl; 

    send_mat_to_host(sub_mttkrp, &debug_mat);
    INFO << "AtB\n" << debug_mat << std::endl; 
#endif

    cublasHandle_t handle2;
    check_cublas(cublasCreate(&handle2), "create cublas handle2");

    double alpha = 1.0;
    double beta = -1.0;

    check_cublas(cublasDgeam(handle2, CUBLAS_OP_N, CUBLAS_OP_N, Y->n_rows, Y->n_cols, 
    &alpha, Y->vals, Y->n_rows, &beta, sub_mttkrp->vals, sub_mttkrp->n_rows, 
    Y->vals, Y->n_rows), "Y <- 1.0 * gram * X - 1.0 * sub_mttkrp");

    cublasDestroy(handle2);

    // Update Y all at once
    fixAbsNumericalError_gpu(X->vals, 1e-12, 0.0, X->n_cols * X->n_rows);
    fixAbsNumericalError_gpu(Y->vals, 1e-12, 0.0, Y->n_cols * Y->n_rows);

#if DEBUG==0
    // send_mat_to_host(X, &debug_mat);
    // INFO << "after numerical error fix: X\n" << debug_mat << std::endl; 
    // // Verified
    send_mat_to_host(Y, &debug_mat);
    INFO << "after numerical error fix: Y\n" << debug_mat << std::endl; 
#endif

    // NotOptMask.fill(1);
    // NotOptMask.rowwise_mult(NotOptCols);

    // // Check if below operations are supported
    NonOptSet = (*Y < 0) % (PassiveSet == 0);
    InfeaSet = (*X < 0) % PassiveSet;

    NotGood = NonOptSet.sum() + InfeaSet.sum();

    cudaMemcpy(debug_uvec.memptr(), NotGood.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
    INFO << "NotGood--inner\n" << debug_uvec << "\n";

    NotOptCols = NotGood > 0;
    numNonOptCols = NotOptCols.sum();
    printf("numNonOptCols.sum(): %d\n", numNonOptCols);
    
    // send_umat_to_host(&PassiveSet, &debug_umat);
    // INFO << "PassiveSet\n" << debug_umat << "\n";

    // cudaMemcpy(debug_uvec.memptr(), P.vals, sizeof(uint32_t) * P.size, cudaMemcpyDeviceToHost);
    // INFO << "P\n" << debug_uvec << "\n";

    // cudaMemcpy(debug_uvec.memptr(), Ninf.vals, sizeof(uint32_t) * Ninf.size, cudaMemcpyDeviceToHost);
    // INFO << "Ninf\n" << debug_uvec << "\n";
  }


  if (!success) {
    ERR << "BPP failed, check Cols3" << std::endl;
    exit(1);
  }
  free_mat_gpu(fm_gpu);
  free_mat_gpu(mttkrp_t_gpu);
  free_mat_gpu(gram_gpu);
  free_mat_gpu(sub_mttkrp);
  free_mat_gpu(X);
  free_mat_gpu(Y);
  }

}
// Checks functionality for operator overloading
// and sum / accum operations
/*
int main(int argc, char* argv[]) {
  INFO << "Investigating BPP..." << std::endl;
  arma::arma_rng::set_seed(13);

  MAT fm = arma::randn<MAT>(NROW, RANK);

  MAT gram = fm.t() * fm;

  cudaError_t cudaStatus = cudaSetDevice(2);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
    return 1;
  }

  MAT_GPU * fm_gpu = send_mat_to_gpu(&fm);
  MAT_GPU * gram_gpu = send_mat_to_gpu(&gram);

  MAT debug_mat = arma::zeros<MAT>(fm_gpu->n_rows, fm_gpu->n_cols);
  MAT _debug_mat = arma::zeros<MAT>(RANK, RANK);

  send_mat_to_host(fm_gpu, &debug_mat);
  send_mat_to_host(gram_gpu, &_debug_mat);
  INFO << debug_mat << std::endl;
  INFO << _debug_mat << std::endl;


  UUMAT debug_umat = arma::zeros<UUMAT>(fm_gpu->n_rows, fm_gpu->n_cols);
  UUVEC debug_uvec = arma::zeros<UUVEC>(fm_gpu->n_cols);

  for (int n = 0; n < 10; ++n) {
    UMAT_GPU test = *fm_gpu > (-1 + 0.35 * n);
    UVEC_GPU sum = test.sum();
    UVEC_GPU NonOptCols = sum > 0;
    unsigned int numNonOptCols = NonOptCols.sum();

    send_umat_to_host(&test, &debug_umat);
    cudaMemcpy(debug_uvec.memptr(), sum.vals, sizeof(uint32_t) * sum.size, cudaMemcpyDeviceToHost);

    INFO << debug_umat << std::endl;
    INFO << debug_uvec << std::endl;
    INFO << numNonOptCols << std::endl;
  }
  
  free_mat_gpu(fm_gpu);
}
*/

/*
int main(int argc, char* argv[]) {
  INFO << "Investigating BPP..." << std::endl;
  arma::arma_rng::set_seed(13);
  MAT fm = arma::randn<MAT>(NROW, RANK);
  // MAT gram = arma::zeros<MAT>(RANK, RANK);
  // arma::arma_rng::set_seed(12);
  MAT mttkrp = arma::randn<MAT>(RANK, NROW);

  MAT gram = fm.t() * fm;

  int rank = RANK;
  int col_start_idx = 1;
  int col_end_idx = 12;
  int nrhs = col_end_idx - col_start_idx + 1;

  INFO << "==== CPU ====" << std::endl;
  INFO << gram << std::endl;
  BPPNNLS<MAT, VEC> subProblem(gram, (MAT)mttkrp.cols(col_start_idx, col_end_idx), true);

  subProblem.solveNNLS();
  INFO << subProblem.getSolutionMatrix() << std::endl;

  INFO << "==== GPU ====" << std::endl;
  cudaError_t cudaStatus = cudaSetDevice(2);
  if (cudaStatus != cudaSuccess) {
      std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
      return 1;
  }

  // Reset the CUDA device
  // cudaDeviceReset();

  // // Check for errors after reset
  // cudaStatus = cudaGetLastError();
  // if (cudaStatus != cudaSuccess) {
  //     std::cerr << "cudaDeviceReset failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
  //     return 1;
  // }

  // std::cout << "CUDA device reset successfully." << std::endl;
  
  MAT_GPU * mttkrp_gpu = send_mat_to_gpu(&mttkrp);
  MAT_GPU * gram_gpu = send_mat_to_gpu(&gram);

  // check_cuda(cudaDeviceSynchronize(), "ssu");
  
  MAT_GPU * sub_mttkrp = mttkrp_gpu->cols(col_start_idx, col_end_idx);
  MAT_GPU * Y = init_mat_gpu(sub_mttkrp->n_rows, sub_mttkrp->n_cols); // res??
  
  MAT_GPU * X = init_mat_gpu(sub_mttkrp->n_rows, sub_mttkrp->n_cols); // solution matrix
  cudaMemset(X->vals, 0.0, sizeof(double) * X->n_rows * X->n_cols);

  mat_mat_mul(gram_gpu->vals, X->vals, Y->vals, gram_gpu->n_rows, gram_gpu->n_cols, X->n_cols, 1.0, 0.0);

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");

  double alpha = 1.0;
  double beta = -1.0;

  check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Y->n_rows, Y->n_cols, 
  &alpha, Y->vals, Y->n_rows, &beta, sub_mttkrp->vals, sub_mttkrp->n_rows, 
  Y->vals, Y->n_rows), "Y <- 1.0 * Y - 1.0 * sub_mttkrp");

  cublasDestroy(handle);

  MAT debug = arma::zeros<MAT>(X->n_rows, X->n_cols);
  UUMAT debug_umat = arma::zeros<UUMAT>(X->n_rows, X->n_cols);
  UUVEC debug_uvec = arma::zeros<UUVEC>(X->n_cols);

  send_mat_to_host(Y, &debug);

  INFO << "Y\n" << debug << "\n";

  UMAT_GPU PassiveSet = *X > 0;
  send_umat_to_host(&PassiveSet, &debug_umat);

  INFO << "PassiveSet\n" << debug_umat << "\n";

  UMAT_GPU LO = *Y < 0;
  UMAT_GPU RO = PassiveSet == 0;
  
  UMAT_GPU negX = *X < 0;
  
  UMAT_GPU NonOptSet = LO % RO;
  UMAT_GPU InfeaSet = negX % PassiveSet;

  send_umat_to_host(&NonOptSet, &debug_umat);
  INFO << "NonOptSet\n" << debug_umat << "\n";
  send_umat_to_host(&InfeaSet, &debug_umat);
  INFO << "InfeaSet\n" << debug_umat << "\n";


  UVEC_GPU SumNonOptSet = NonOptSet.sum();
  UVEC_GPU SumInfeaSet = InfeaSet.sum();

  UVEC_GPU NotGood = SumNonOptSet + SumInfeaSet; // need to add InfeaSet.sum();
  cudaMemcpy(debug_uvec.memptr(), NotGood.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
  INFO << "NotGood\n" << debug_uvec << "\n";

  UVEC_GPU NotOptCols = NotGood > 0;
  cudaMemcpy(debug_uvec.memptr(), NotOptCols.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
  INFO << "NotOptCols\n" << debug_uvec << "\n";

  // INFO << "numNonOptCols: " << NotGood.sum() << std::endl;
  unsigned int numNonOptCols = NotOptCols.sum();
  
  bool success = false;
  unsigned int iter = 0;
  unsigned int MAX_ITERATIONS = NROW * 5;

  while (numNonOptCols > 0) {
    printf("numNonOptCols: %d\n", numNonOptCols);
    iter++;

    if (iter > 5) {
      success = false;
      break;
    }
    // Identify Col1, Col2, Col3
    
    // Update X and Y


    // Apply AbsNumericalError for both X and Y

    // Update NonOptSet
    LO = *Y < 0;
    RO = PassiveSet == 0;
    NonOptSet = LO % RO;

    negX = *X < 0;
    InfeaSet = negX % PassiveSet;

    SumNonOptSet = NonOptSet.sum();
    SumInfeaSet = InfeaSet.sum();
    NotGood = SumNonOptSet + SumInfeaSet; // need to add InfeaSet.sum();

    cudaMemcpy(debug_uvec.memptr(), NotGood.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
    INFO << "_NotGood - inside while\n" << debug_uvec << "\n";

    NotOptCols = (NotGood > 0);

    cudaMemcpy(debug_uvec.memptr(), NotOptCols.vals, sizeof(uint32_t) * NotOptCols.size, cudaMemcpyDeviceToHost);
    INFO << "_NotOptCols - inside while\n" << debug_uvec << "\n";

    numNonOptCols = NotOptCols.sum();
    INFO << "_numNonOptCols " << numNonOptCols << "\n";
  }
}
*/

int main_aoadmm(int argc, char* argv[]) {
  INFO << "Profiling cholesky implementations..." << std::endl;
  arma::arma_rng::set_seed(103);
  MAT fm = arma::randu<MAT>(NROW, RANK);
  MAT mttkrp = arma::randu<MAT>(NROW, RANK);
  MAT mttkrp_t = mttkrp.t();
  MAT gram = fm.t() * fm;
  
  INFO << mttkrp << std::endl;

  MAT L = arma::chol(gram, "lower");
  MAT Lt = L.t();
  INFO << L << std::endl;
  
  MAT X = arma::solve(arma::trimatl(L), mttkrp_t);
  MAT ans = arma::solve(arma::trimatu(Lt), X);

  // INFO << ans << std::endl;

  INFO << ans << std::endl;

  MAT_GPU * gram_gpu = send_mat_to_gpu(&gram);
  MAT_GPU * mttkrp_gpu = send_mat_to_gpu(&mttkrp_t);

  mat_cholesky_gpu(gram_gpu, true);
  copy_mat_gpu_lower(gram_gpu, gram_gpu);
  cudaDeviceSynchronize();
  mat_cholesky_solve_gpu(gram_gpu, mttkrp_gpu, true);

  MAT cholesky_from_gpu = MAT(RANK, RANK);
  cudaMemcpy(cholesky_from_gpu.memptr(), gram_gpu->vals, sizeof(double) * RANK * RANK, cudaMemcpyDeviceToHost);
  INFO << cholesky_from_gpu << std::endl;

  MAT result_from_gpu = MAT(RANK, NROW);
  cudaMemcpy(result_from_gpu.memptr(), mttkrp_gpu->vals, sizeof(double) * NROW * RANK, cudaMemcpyDeviceToHost);
  
  INFO << result_from_gpu << std::endl;
  INFO << gram << std::endl;
  INFO << gram * result_from_gpu << std::endl;
}

int main_mttkrp(int argc, char* argv[]) {
  if (argc < 2) {
    INFO << "Usage: " << argv[0] << " <full_path_to_input_tensor>" << std::endl;
    return 1;
  }

  INFO << "Profiling sparse mttkrp implementations..." << std::endl;
  std::string filename = argv[1];
  // std::string filename = "/home/users/ypsoh/hpctensor/synthetic/large_sparse_tensor.tns";oo
  // std::string filename = "/home/users/ypsoh/hpctensor/data/flickr-4d.tns";
  // std::string filename = "/home/users/ypsoh/hpctensor/data/uber.tns";
  int num_iters = 1;
  
  {
    double wtime = omp_get_wtime();
    planc::SparseTensor t(filename);
    printf("tensor reading took %f s\n", omp_get_wtime()-wtime);
    planc::NCPFactors fms(t.dimensions(), RANK, false);
    
    fms.normalize();
    
    MAT* o_mttkrp = new MAT[t.modes()];
    for (int m = 0; m < t.modes(); ++m) {
      o_mttkrp[m].zeros(RANK, t.dimensions(m));
    }
    
    int num_iters = 1;
    for (int n = 0; n < num_iters; ++n) {
      for (int m = 0; m < t.modes(); ++m) {
        wtime = omp_get_wtime();
        t.mttkrp(m, fms.factors(), &o_mttkrp[m]);
        printf("mode: %d\t time: %f\n", m, omp_get_wtime()-wtime);
      }
    }
  }
  
  
  {
    planc::ALTOTensor<LIType> at(filename);
    planc::NCPFactors fms(at.dimensions(), RANK, false);
    
    fms.normalize();
    
    double wtime;

    MAT* o_mttkrp = new MAT[at.modes()];
    for (int m = 0; m < at.modes(); ++m) {
      o_mttkrp[m].zeros(RANK, at.dimensions(m));
    }
    
    for (int n = 0; n < 1; ++n) {
      for (int m = 0; m < at.modes(); ++m) {
        wtime = omp_get_wtime();
        at.mttkrp(m, fms.factors(), &o_mttkrp[m]);
        printf("[ALTO] mode: %d\t time: %f\n", m, omp_get_wtime()-wtime);
        printf("[ALTO] norm of output -- mode: %d: %f\n", m, arma::norm(o_mttkrp[m], "fro"));
      }
    }
  }
  
  
  {
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    planc::BLCOTensor<LIType> bt(filename);
    planc::NCPFactors fms(bt.dimensions(), RANK, false);
    
    fms.normalize();
    
    double wtime_s, wtime;

    MAT* o_mttkrp = new MAT[bt.modes()];
    for (int m = 0; m < bt.modes(); ++m) {
      o_mttkrp[m].zeros(RANK, bt.dimensions(m));
    }
    
    for (int n = 0; n < num_iters; ++n) {
      for (int m = 0; m < bt.modes(); ++m) {
        wtime = omp_get_wtime();
        bt.mttkrp(m, fms.factors(), &o_mttkrp[m]);
        printf("[BLCO] mode: %d\t time: %f\n", m, omp_get_wtime()-wtime);
        printf("[BLCO] norm of output -- mode: %d: %f\n", m, arma::norm(o_mttkrp[m], "fro"));
      }
    }
    // 
  }  
}