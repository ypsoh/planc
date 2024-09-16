
#ifndef COMMON_SPARSE_TENSOR_HPP_
#define COMMON_SPARSE_TENSOR_HPP_

#ifdef MKL_FOUND
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <armadillo>
#include <fstream>
#include <ios>
#include <random>
#include <string>
#include <type_traits>

#include <utility>
#include <vector>
#include <unordered_map>
#include <cmath>
#include "common/utils.h"
#include "common/ncpfactors.hpp"

namespace planc {

class SparseTensor {
  protected:
    UVEC m_dimensions;
    UWORD m_numel;
    unsigned int rand_seed;
  public:
    int m_modes;
    std::vector<double> m_data;
    std::vector<std::vector<int>> m_indices;

    // compact indices and mappings are required for
    // efficient representation of the indices for Sparse TF
    std::vector<std::vector<int>> m_compact_indices;
    std::vector<std::unordered_map<int, int>> m_mappings;

    // Threadlocks that are used for sparse mttkrp
    // instantiated once the longest mode is identified
    // this is so that we can 'fine-grain' control when a output row of mttkrp
    // is being updated without contention
    mutable std::vector<omp_lock_t> m_locks;

    SparseTensor() {
      this->m_modes = 0;
      this->m_numel = 0;
    }

    SparseTensor(std::string filename): m_numel(0), m_modes(0) {
      
      std::cout << "Reading tensor from " << filename << std::endl;
      std::ifstream ifs(filename, std::ios_base::in);
      std::string line;

      int nmodes = 0;
      std::string element;

      // ifs.seekg(0); // go back to beginning of file to read non zeros

      // first time reading tensor, traverse to end to 
      // 1. count number of modes 2. count nnzs
      UWORD _nnz = 0;
      while (std::getline(ifs, line, '\n')) {
        if (nmodes == 0) {
          std::stringstream _line(line);
          while(_line >> element) {
            nmodes++;
          }
          nmodes--; // since the last element is the value not coordinate
          this->m_modes = nmodes;
        }
        _nnz++;
      }
      m_numel = _nnz;
      INFO << "num elements: " << m_numel << "\n";

      // set vector size to predetermined size
      this->m_indices.resize(this->m_modes);
      for (int m = 0; m < this->m_modes; ++m) {
        this->m_indices[m].resize(this->m_numel);
      }
      this->m_data.resize(this->m_numel);

      // go back to beginning to parse idx and values
      ifs.clear();
      ifs.seekg(0);

      unsigned long long nnz_idx = 0;

      while (std::getline(ifs, line, '\n')) {
        int mode_idx = 0;
        char * ptr = &line[0];
        for (int m = 0; m < this->m_modes; ++m) {
          this->m_indices[m][nnz_idx] = (int) strtol(ptr, &ptr, 10);
        }
        this->m_data[nnz_idx] = (double) strtod(ptr, &ptr);
        ++nnz_idx;
      }

      // Map the 'raw' indices of the non zeros to a
      // corresponding row in a factor matrix starting from 0
      map_to_compact_indices(false);
      
      // instantiate the omp_lock_ts
      m_locks.resize(longest_mode());
      for (auto lock : m_locks) {
        omp_init_lock(&lock);
      }
    }

    ~SparseTensor() {}
    /*
    Used to map original indices to compact indices
    (e.g. (34, 32, 53, 32) --> (0, 1, 2, 1))
    */
    void map_to_compact_indices(bool do_remap) {
      this->m_dimensions = UVEC(this->m_modes);
      this->m_compact_indices.resize(this->m_modes);
      this->m_mappings.resize(this->m_modes);

      // Remapping is usually needed for real datasets where there is no guarantee 
      // all indices will be occupied, the "remapping -- if(false)" is due to the 
      // the 0 or 1 offset issue
      if (do_remap) {
        for (int m = 0; m < this->m_modes; ++m) {
          int new_index = 0;
          for (int index: this->m_indices[m]) { // original indices
            if (this->m_mappings[m].find(index) == this->m_mappings[m].end()) {
              this->m_mappings[m][index] = new_index++;
            }
            this->m_compact_indices[m].push_back(this->m_mappings[m][index]);
          }
        }
        for (int m = 0; m < this->m_modes; ++m) {
          this->m_dimensions[m] = this->m_mappings[m].size();
        }
      }
      else {
        for (int m = 0; m < this->m_modes; ++m) {
          // find scope
          int max_idx = arma::max(arma::Col<int>(this->m_indices[m]));
          int min_idx = arma::min(arma::Col<int>(this->m_indices[m]));
          int scope = max_idx - min_idx;
          for (int index: this->m_indices[m]) {
            this->m_compact_indices[m].push_back(index-1);
          };
          this->m_dimensions[m] = max_idx;
        }
      }
    }
    // TODO: Will implement once basic Sparse TF is done
    // or maybe just yield the mappings info as separate output for
    // actually interpreting the factor matrices
    void restore_index_mapping() {}
    int modes() const { return m_modes; }
    UWORD numel() const { return m_numel; }
    UVEC dimensions() const { return m_dimensions; }
    unsigned int dimensions(int m) const { return m_dimensions[m]; }
    int longest_mode() const { return arma::max(m_dimensions); }

    /**
     * @brief Computes rel. error between input tensor X and kruskal model M
     * || X - M ||^2 / || X ||^2
     * For sparse tensor formats, assumes mttkrp_mat is same dims as factor matrix
     * This is different from dense TF
     * @param[in] factors NCPFactors - factor matrices
     * @return double the squared error in respect to input tensor
     */
    virtual double err(planc::NCPFactors &factors, const MAT &i_mttkrp_mat, const int mode) const {
      double inner_prod = 0;
      unsigned int rank = factors.rank();
      double accum[rank] = {}; // init to zero

      int last_mode_dim = factors.dimension(mode);

      #pragma omp parallel for schedule(static)
      for (int j = 0; j < rank; ++j) {
        for (int i = 0; i < last_mode_dim; ++i) {
          accum[j] += factors.factor(mode)(i, j) * i_mttkrp_mat(i, j);
        }
      }
      for (int i = 0; i < rank; ++i) {
        inner_prod += accum[i] * factors.lambda()[i];
      }

      double norm_x = 0;
      // Compute norm(X)^2
      #pragma omp parallel for reduction(+:norm_x) schedule(static)
      for(int i = 0; i < m_numel; i++) {
        norm_x += m_data[i] * m_data[i];
      }

      // Compute norm of factor matrices
      // create gram matrix
      MAT tmp_gram = arma::ones<MAT>(rank, rank);

      // compute the hadamard of the factor grams
      factors.gram(&tmp_gram);

      #pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < i+1; ++j) {
          tmp_gram(i, j) *= factors.lambda()[i] * factors.lambda()[j];
        }
      }

      double norm_u = 0;
      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < i; ++j) {
          norm_u += tmp_gram(i, j) * 2;
        }
        norm_u += tmp_gram(i, i);
      }
      norm_u = std::abs(norm_u);

      double norm_residual = norm_x + norm_u - 2 * inner_prod;
      // if (norm_residual > 0.0) {
      norm_residual = std::sqrt(norm_residual);
      // }
      INFO << "norm_x: " << norm_x << "inner: " << inner_prod << "norm_u: " << norm_u << std::endl;
      double rel_fit_err = (norm_residual/std::sqrt(norm_x));
      return rel_fit_err;
    }

    double norm() const {
      double norm_fro = 0;
      for (int i = 0; i < this->m_numel; ++i) {
        norm_fro += (this->m_data[i] * this->m_data[i]);
      }
      return norm_fro;
    }
    
    void print() const {
      INFO << "Dimensions: ";
      for (int m = 0; m < this->m_modes; ++m) {
        INFO << " " << this->m_dimensions[m];
      }
      INFO << std::endl;

      INFO << "Number of non-zeros: " << this->m_numel << std::endl;;
      INFO << "Sparsity: " << (double)this->m_numel / arma::prod(this->m_dimensions) << std::endl;;
    }

    virtual void mttkrp(const int i_n, MAT *i_factors, MAT *o_mttkrp) const {
      (*o_mttkrp).zeros();

      unsigned int rank = i_factors[i_n].n_cols;
      int max_threads = omp_get_max_threads();

      // Do awesome mttkrp...
      double rows[max_threads][rank];

      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        double * row = rows[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < m_numel; ++i) {
          for(int r = 0; r < rank; ++r) {
            row[r] = this->m_data[i]; // init temp accumulator
          }
          // calculate mttkrp for current non-zero
          for (int m = 0; m < m_modes; ++m) {
            if (m != i_n) {
              unsigned int row_id = m_compact_indices[m][i];
              // unsigned int row_id = 0;
              for (int r = 0; r < rank; ++r) {
                row[r] *= i_factors[m](row_id, r);
              }
            }
          }

          // update destination row
          unsigned int dest_row_id = m_compact_indices[i_n][i];
          omp_set_lock(&(m_locks[dest_row_id]));
          for (int r = 0; r < rank; ++r) {
            o_mttkrp->at(r, dest_row_id) += row[r];
          }
          omp_unset_lock(&(m_locks[dest_row_id]));
        } // for each non-zero
      } // #pragma omp parallel
      printf("norm of first mttkrp output -- mode: %d: %f\n", i_n, arma::norm(*o_mttkrp, "fro"));
      // exit(0);
    }
};
}

#endif  // COMMON_SPARSE_TENSOR_HPP_
