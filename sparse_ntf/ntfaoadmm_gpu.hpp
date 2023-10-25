#ifndef NTF_NTFAOADMM_GPU_HPP_
#define NTF_NTFAOADMM_GPU_HPP_

#include "sparse_ntf/auntf_gpu.hpp"
#include "cuda_utils.h"
#include "aoadmm.h"

namespace planc {
template <class T>
class NTFAOADMM_GPU : public AUNTF_GPU <T> {
  private:
    MAT_GPU** aux_factors_gpu;
    int admm_iter;
    double tolerance;

  protected:
    void update_gpu(const int mode, const MAT_GPU * gram, MAT_GPU ** factors, MAT_GPU * o_mttkrp) {
      aoadmm_update(factors[mode], aux_factors_gpu[mode], o_mttkrp, gram, admm_iter, tolerance);
      // exit(0);
    }

  public:
    NTFAOADMM_GPU(const T &i_tensor, const int i_k, algotype i_algo)
      : AUNTF_GPU<T>(i_tensor, i_k, i_algo) {
        // set up dual variable that only exists in GPU
        aux_factors_gpu = new MAT_GPU*[i_k];

        for (int m = 0; m < i_tensor.modes(); ++m) {
          int n_rows = i_tensor.dimensions()[m];
          int n_cols = i_k;
          printf("aux_factors_gpu[%d]: (%dx%d)\n", m, n_rows, n_cols);
          aux_factors_gpu[m] = init_mat_gpu(n_rows, n_cols);
        }
        admm_iter = 5;
        tolerance = 0.01;
      }
};  // class NTFAOADMM_GPU
}  // namespace planc

#endif  // NTF_NTFAOADMM_GPU_HPP_
