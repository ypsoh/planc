#include "aoadmm.h"

void aoadmm_update(
  MAT_GPU * fm, MAT_GPU * aux_fm, MAT_GPU * o_mttkrp_gpu, 
  const MAT_GPU * gram, int admm_iter, double tolerance) {
  bool stop_iter = false;

  int m = fm->n_rows;
  int n = fm->n_cols;

  double _alpha = mat_trace_gpu(gram) / n;
  _alpha = (_alpha > 0) ? _alpha : 0.01;

  MAT_GPU * tempgram = init_mat_gpu(n, n);
  copy_mat_gpu(tempgram, gram);  
  mat_add_diag_gpu(tempgram, _alpha);

  MAT_GPU * L = init_mat_gpu(n, n);
  copy_mat_gpu(L, tempgram);
  mat_cholesky_gpu(L); // L is factorized
  // mat_cholesky_solve_gpu(L, fm, );
  MAT_GPU * prev_fac = init_mat_gpu(m, n);
  MAT_GPU * update_fm = init_mat_gpu(m, n);
  MAT_GPU * aux_fm_t = init_mat_gpu(n, m);
  
  copy_mat_gpu(update_fm, fm);
  copy_mat_gpu(prev_fac, fm);

  // For gpu kernel calls
  int num_tblocks = (m * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int fm_size = m * n;
  int gram_size = n * n;

  //--- DEBUG compute norms and stuff delete later
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");

  double debug;

  // cudaFree(debug);

  double alpha, beta;
  for (int i = 0; i < admm_iter && !stop_iter; ++i) {
    copy_mat_gpu(prev_fac, update_fm);
    //----- aux_fm_t = 1.0 * updated_fm.T + 1.0 * aux_fm.T
    //----- aux_fm_t = mttkrp + alpha * aux_fm_t
    alpha = 1.0; beta = 1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, 
      &alpha, update_fm->vals, m, &beta, aux_fm->vals, m, 
      aux_fm_t->vals, n), "aux_fm_t <- 1.0 * updated_fm.T + 1.0 * aux_fm.T");

    beta = _alpha;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 
      &alpha, o_mttkrp_gpu->vals, n, &beta, aux_fm_t->vals, n, 
      aux_fm_t->vals, n), "aux_fm_t <- 1.0 * mttkrp + alpha * aux_fm_t");

    // cudaMemcpy(dd, aux_fm_t->vals, sizeof(double) * 16, cudaMemcpyDeviceToHost);
    // for (int nn = 0; nn < 16; ++nn) printf("%f\n", dd[nn]);
    // exit(0);
    mat_cholesky_solve_gpu(L, aux_fm_t, true);
    cudaDeviceSynchronize();

    beta = -1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, 
      &alpha, aux_fm_t->vals, n, &beta, aux_fm->vals, m, 
      update_fm->vals, m), "updated_fm <- 1.0 * aux_fm_t.T - 1.0 * aux_fm");

    __apply_threshold<<<num_tblocks, BLOCK_SIZE>>>(update_fm->vals, fm_size, 0.0, 0.0);
    cudaDeviceSynchronize();

    beta = 1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 
      &alpha, aux_fm->vals, m, &beta, update_fm->vals, m, 
      aux_fm->vals, m), "aux_fm <- 1.0 * aux_fm + 1.0 * updated_fm");

    beta = -1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, 
      &alpha, aux_fm->vals, m, &beta, aux_fm_t->vals, n, 
      aux_fm->vals, m), "aux_fm <- 1.0 * aux_fm - 1.0 * aux_fm_t.T");

    // check_cublas(cublasDnrm2(handle, fm_size, update_fm->vals, 1, &debug), "cublas Dnrm2");
    // printf("after th %f\n", debug);

    // aux_fm = aux_fm + updated_fac - aux_fm_t

    //--- Compute fm norm and dual variable norm    
    double fnorm;
    double dnorm;

    check_cublas(cublasDnrm2(handle, fm_size, update_fm->vals, 1, &fnorm), "cublas Dnrm2");
    check_cublas(cublasDnrm2(handle, fm_size, aux_fm->vals, 1, &dnorm), "cublas Dnrm2");

    //--- END

    //--- Compute f_res and d_res    
    double fres;
    double dres;

    double * diff; // used for diffs
    cudaMalloc((void**)&diff, sizeof(double) * fm_size);
    cudaMemcpy(diff, update_fm->vals, sizeof(double) * fm_size, cudaMemcpyDeviceToDevice);
    dvec_sub(diff, prev_fac->vals, diff, fm_size);
    check_cublas(cublasDnrm2(handle, fm_size, diff, 1, &fres), "cublas Dnrm2");

    beta = -1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, 
      &alpha, update_fm->vals, m, &beta, aux_fm_t->vals, n, 
      aux_fm_t->vals, n), "aux_fm_t <- 1.0 * update_fm.T - 1.0 * aux_fm_t");
    check_cublas(cublasDnrm2(handle, fm_size, aux_fm_t->vals, 1, &dres), "cublas Dnrm2");

    // printf("fnorm: %f dnorm: %f fres: %f dres: %f\n", fnorm, dnorm, fres, dres);
    cudaFree(diff);
    //--- END

    // Check convergence
    if (dres < (tolerance * fnorm) && fres < (tolerance * dnorm)) {
      stop_iter = true;
    }
  } // end of admm iteration

  // Update fm
  copy_mat_gpu(fm, update_fm);
  cudaDeviceSynchronize();
  
  //--- normalize dual variable aux_fm
  double * temp_lambda; 
  cudaMalloc((void**)&temp_lambda, sizeof(double) * n); // dummy lambda
  
  normalize_fm(aux_fm, temp_lambda);
  cudaFree(temp_lambda);
  
  // check_cuda(cudaGetLastError(), "normalize aux_fm");

  //---

  free_mat_gpu(prev_fac);
  free_mat_gpu(update_fm);
  free_mat_gpu(aux_fm_t);

  free_mat_gpu(L);
  free_mat_gpu(tempgram);

  cublasDestroy(handle);
}
