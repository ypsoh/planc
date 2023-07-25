
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
#include "common/utils.h"

namespace planc {

class SparseTensor {
  private:
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

    SparseTensor() {
      this->m_modes = 0;
      this->m_numel = 0;
    }

    SparseTensor(std::string filename): m_numel(0) {
      std::cout << "Reading tensor from " << filename << std::endl;

      std::ifstream ifs(filename, std::ios_base::in);
      std::string line;

      int nmodes = 0;
      int* dims = NULL;
      std::string element;

      ifs.seekg(0); // go back to beginning of file to read non zeros

      while (std::getline(ifs, line, '\n')) {
        // std::cout << line << ""
        // for every line

        std::stringstream _line(line);

        if (nmodes == 0) {
          while(_line >> element) {
            nmodes++;
          }
          nmodes--; // since the last element is the value not coordinate
          this->m_modes = nmodes;

          // set the coodrinate vectors
          for (int i = 0; i < nmodes; ++i) {
            std::vector<int> mode_indices;
            this->m_indices.push_back(std::vector<int>());
            // required for mapping raw indices to compact indices
            this->m_compact_indices.push_back(std::vector<int>());
          }
          ifs.seekg(0);
          continue;
        }
        int index = 0;
        while (_line >> element) {
          if (index < nmodes) {
            this->m_indices[index].push_back(std::stoi(element));
          } else {
            this->m_data.push_back(std::stod(element));
          }
          index++;
        }
        this->m_numel++;
      }
    }

    ~SparseTensor() {}

    /*
    Used to map original indices to compact indices
    (e.g. (34, 32, 53, 32) --> (0, 1, 2, 1))
    */
    void map_to_compact_indices() {
      this->m_dimensions = UVEC(this->m_modes);

      for (int m = 0; m < this->m_modes; ++m) {
        this->m_compact_indices.push_back(std::vector<int>());
        this->m_mappings.push_back(std::unordered_map<int, int>());
      }

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
    // Will implement once basic Sparse TF is done
    void restore_index_mapping() {}
    int modes() const { return m_modes; }
    UWORD numel() const { return m_numel; }
    UVEC dimensions() const { return m_dimensions; }
    int dimensions(int m) const { return m_dimensions[m]; }
    /**
     * @brief Computes error between input tensor and b
     * However, we need to decide how to compute in case of
     * dealing with sparse tensors (or tensors)
     *
     * @param[in] b an input tensor
     * @return double the squared error in respect to input tensor
     */
    double err(const Tensor &b) const {
      double norm_fro = 0;
      double err_diff;
      for (int i = 0; i < this->m_numel; ++i) {
        err_diff = this->m_data[i] - b.m_data[i];
        norm_fro += err_diff * err_diff;
      }
      return norm_fro;
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
    void mttkrp(const int i_n, const MAT &i_krp, MAT *o_mttkrp) const {
      (*o_mttkrp).zeros();
      // Do awesome mttkrp...

    }
};
}

#endif  // COMMON_SPARSE_TENSOR_HPP_
