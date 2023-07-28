#ifndef COMMON_ALTO_TENSOR_HPP_
#define COMMON_ALTO_TENSOR_HPP_

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
#include "common/bitops.hpp"

#define MAX_NUM_MODES 5
#define MIN_FIBER_REUSE 4

typedef enum PackOrder_ { LSB_FIRST, MSB_FIRST } PackOrder;
typedef enum ModeOrder_ { SHORT_FIRST, LONG_FIRST, NATURAL } ModeOrder;

typedef struct Inverval_ {
  int start;
  int stop;
} Interval;

struct MPair {
  int mode;
  int bits;
};

namespace planc {

template <typename LIT>
class ALTOTensor : public SparseTensor {
  public:
    int m_num_partitions = 0;

    // ALTO stuff
    LIT alto_mask = 0;
    std::vector<LIT> mode_masks;
    LIT *idx = nullptr;

    std::vector<double> m_alto_data;
    std::vector<LIT> m_alto_indices; // linearized

    std::vector<int> m_partition_ptr;
    std::vector<std::vector<Interval>> m_partition_intervals;

    // for optimized ALTO performance
    std::vector<std::vector<double>> m_output_fibers;

    // needed for conflict resolution ALTO implementation
    std::vector<LIT> alto_cr_masks;
    LIT *cr_masks = nullptr;

    ALTOTensor(std::string filename) : SparseTensor(filename) {
      //uint64_t ticks;
      double wtime_s, wtime;

      // Do ALTO stuff
      INFO << "Generating ALTO tensor" << std::endl;
      // ALTOTensor<LIT> * _at;
      this->mode_masks.resize(m_modes, 0);

      UWORD numel = this->numel();
      m_alto_data.resize(numel);
      m_alto_indices.resize(numel);

      wtime_s = omp_get_wtime();
      setup_packed_alto(LSB_FIRST, SHORT_FIRST);
      wtime = omp_get_wtime() - wtime_s;
      printf("ALTO: setup time = %f (s)\n", wtime);


      LIT ALTO_MASKS[MAX_NUM_MODES];
      for (int n = 0; n < m_modes; ++n) {
        ALTO_MASKS[n] = mode_masks[n];
      }

      m_alto_data.resize(numel);
      m_alto_indices.resize(numel);

      // Linearization
      wtime_s = omp_get_wtime();
      #pragma omp parallel for
      for (int i = 0; i < numel; i++) {
          LIT alto = 0;
          m_alto_data[i] = m_data[i];
          for (int j = 0; j < m_modes; j++) {
              alto |= pdep(static_cast<unsigned long long>(m_compact_indices[j][i]), ALTO_MASKS[j]);
          }
          m_alto_indices[i] = alto;

// for debugging
  #if 0
          for (int j = 0; j < m_modes; j++) {
              int mode_idx = 0;
              mode_idx = pext(alto, ALTO_MASKS[j]);
              assert(mode_idx == m_compact_indices[j][i]);
          }
  #endif
      } // end of linearization
      wtime = omp_get_wtime() - wtime_s;
      printf("ALTO: Linearization time = %f (s)\n", wtime);

      //Sort the nonzeros based on their line position.
      wtime_s = omp_get_wtime();
      sort_alto();
      wtime = omp_get_wtime() - wtime_s;
      printf("ALTO: sort time = %f (s)\n", wtime);

      wtime_s = omp_get_wtime();
      partition_workload();
      wtime = omp_get_wtime() - wtime_s;
      printf("ALTO: partition time = %f (s)\n", wtime);

      // create direct access memory (e.g. output fibers)
      // create_direct_access_memory(-1, rank, o_fibs);
    }

    // template <typename LIT>
    void sort_alto() {
      UWORD numel = this->numel();

      std::vector<int> inds;
      std::vector<LIT> temp_inds;
      std::vector<double> temp_val;

      inds.resize(numel);
      temp_inds.resize(numel);
      temp_val.resize(numel);

      #pragma omp parallel for
      for (int i = 0; i < numel; ++i) inds[i] = i;

      std::sort(inds.begin(), inds.end(), [&](int a, int b) {
        return m_alto_indices[a] < m_alto_indices[b];
      });
      #pragma omp parallel for
      for (int i = 0; i < numel; i++) temp_inds[i] = m_alto_indices[inds[i]];
      #pragma omp parallel for
      for (int i = 0; i < numel; i++) m_alto_indices[i] = temp_inds[i];
      #pragma omp parallel for
      for (int i = 0; i < numel; i++) temp_val[i] = m_alto_data[inds[i]];
      #pragma omp parallel for
      for (int i = 0; i < numel; i++) m_alto_data[i] = temp_val[i];
    }

    // partition
    void partition_workload() {
      m_num_partitions = omp_get_max_threads();
      m_partition_ptr.resize(m_num_partitions+1);
      m_partition_intervals.resize(m_num_partitions);
      for (int p = 0; p < m_num_partitions; ++p) {
        m_partition_intervals[p].resize(m_modes);
      }

      // needed for conflict resolution implementation
      alto_cr_masks.resize(m_modes);

      // roughly how much nnzs per partition
      int nnz_partition = (m_numel + m_num_partitions - 1) / m_num_partitions;
      printf("num_partitions=%d, nnz_per_partition=%llu\n", m_num_partitions, nnz_partition);

      LIT ALTO_MASKS[MAX_NUM_MODES];
      for (int n = 0; n < m_modes; ++n) {
        ALTO_MASKS[n] = mode_masks[n];
      }

      int alto_bits = popcount(alto_mask);
      m_partition_ptr[0] = 0;

      #pragma omp parallel for schedule(static,1) proc_bind(close)
      for (int p = 0; p < m_num_partitions; ++p) {
        int start_i = p * nnz_partition;
        int end_i = (p + 1) * nnz_partition;

        if (end_i > m_numel)
            end_i = m_numel;

        if (start_i > end_i)
            start_i = end_i;

        // partition pointer points to the index where the nnz index ends for that partition
        // e.g. m_partition_ptr[1] indicates the index where 1st partition nnz ends
        m_partition_ptr[p + 1] = end_i;
      }// omp parallel

      // O(storage requirements) for conflict resolution,
      // using dense/direct-access storage,
      // can be computed in constant time from the subspace id. 
      // The code below finds tighter bounds
      // using interval analysis in linear time (where nnz>> nptrn>> m_modes).
      // e.g. m_partition_intervals[0] has the start and stop indices for 1st partition for all modes
      // m_partition_intervals[0][1]: start and stop indices for the 1st partition for 2nd mode
      #pragma omp parallel for schedule(static,1) proc_bind(close)
      for (int p = 0; p < m_num_partitions; ++p) {
          Interval fib[MAX_NUM_MODES];
          for (int n = 0; n < m_modes; ++n) {
            fib[n].start = m_dimensions(n);
            fib[n].stop = 0;
          }

          // for all elements in the p-th partition
          for (int i = m_partition_ptr[p]; i < m_partition_ptr[p + 1]; ++i) {
            LIT alto_idx = m_alto_indices[i];
            for (int n = 0; n < m_modes; ++n) {
              // retrieve original index for mode n
              int mode_idx = pext(alto_idx, ALTO_MASKS[n]);
              // get the start and stop index for each mode
              // for each 
              fib[n].start = std::min(fib[n].start, mode_idx);
              fib[n].stop = std::max(fib[n].stop, mode_idx);
            }
          }

          for (int n = 0; n < m_modes; ++n) {
              m_partition_intervals[p][n].start = fib[n].start;
              m_partition_intervals[p][n].stop = fib[n].stop;
          }
      }
    }
    // template <typename LIT>
    void create_alto(planc::SparseTensor * sp_tensor, ALTOTensor<LIT> **at, int num_partitions) {

    };

    // template <typename LIT>
    void setup_packed_alto(PackOrder po, ModeOrder mo) {
      LIT ALTO_MASKS[MAX_NUM_MODES] = {};
      int alto_bits_min = 0;
      int alto_bits_max = 0;
      LIT alto_mask = 0;

      int max_num_bits = 0;
      int min_num_bits = sizeof(int) * 8;

      std::vector<MPair> mode_bits(m_modes);

      // initial mode values
      for (int n = 0; n < m_modes; ++n) {
        // this is hard-coded to accomodate 64 bits representation
        int mbits = (sizeof(unsigned long long int) * 8) - clz(m_dimensions(n)-1);
        mode_bits[n].mode = n;
        mode_bits[n].bits = mbits;
        alto_bits_min += mbits;
        max_num_bits = std::max(max_num_bits, mbits);
        min_num_bits = std::min(min_num_bits, mbits);
        printf("num_bits for mode-%d=%d\n", n + 1, mbits);
      }

      alto_bits_max = max_num_bits * m_modes;
      //printf("range of mode bits=[%d %d]\n", min_num_bits, max_num_bits);
      printf("alto_bits_min=%d, alto_bits_max=%d\n", alto_bits_min, alto_bits_max);

      int alto_bits = (int)0x1 << std::max<int>(3, (sizeof(int) * 8) - clz(static_cast<unsigned long long>(alto_bits_min)));
      printf("alto_bits=%d\n", alto_bits);

      double alto_storage = 0;
      alto_storage = m_numel * (sizeof(double) + sizeof(LIT));
      printf("Alto format storage:    %g Bytes\n", alto_storage);

      alto_storage = m_numel * (sizeof(double) + (alto_bits >> 3));
      printf("Alto-power-2 format storage:    %g Bytes\n", alto_storage);

      alto_storage = m_numel * (sizeof(double) + (alto_bits_min >> 3));
      printf("Alto-opt format storage:    %g Bytes\n", alto_storage);

      {//Dilation & shifting.
        int level = 0, shift = 0, inc = 1;

        if (mo == SHORT_FIRST)
          std::sort(mode_bits.begin(), mode_bits.end(), [](MPair& a, MPair& b) { return a.bits < b.bits; });
        else if(mo == LONG_FIRST)
          std::sort(mode_bits.begin(), mode_bits.end(), [](MPair& a, MPair& b) { return a.bits > b.bits; });

        if (po == MSB_FIRST) {
          shift = alto_bits_min - 1;
          inc = -1;
        }

        bool done;
        do {
          done = true;

          for (int n = 0; n < m_modes; ++n) {
            if (level < mode_bits[n].bits) {
              ALTO_MASKS[mode_bits[n].mode] |= (LIT)0x1 << shift;
              shift += inc;
              done = false;
            }
          }
          ++level;
        } while (!done);

        assert(level == (max_num_bits+1));
        assert(po == MSB_FIRST ? (shift == -1) : (shift == alto_bits_min));
      }

      for (int n = 0; n < m_modes; ++n) {
        mode_masks[n] = ALTO_MASKS[n];
        alto_mask |= ALTO_MASKS[n];
        printf("ALTO_MASKS[%d] = 0x%llx\n", n, ALTO_MASKS[n]);
      }
      printf("alto_mask = 0x%llx\n", alto_mask);
    } // end setup_packed_alto
    
    void create_direct_access_memory(int target_mode, int rank, std::vector<std::vector<double>> o_fibers) const {
      {
        double total_storage = 0.0;
        #pragma omp parallel for reduction(+: total_storage) proc_bind(close)
        for (int p = 0; p < m_num_partitions; p++) {
          int num_fibs = 0;
          if (target_mode == -1) {
            // default mode - allocate enough da_mem for all modes with reuse > threshold
            for (int n = 0; n < m_modes; ++n) {
              int fib_reuse = m_numel / dimensions(n);
              if (fib_reuse > MIN_FIBER_REUSE) {
                Interval const intvl = m_partition_intervals[p][n];
                int const mode_fibs = intvl.stop - intvl.start + 1;
                num_fibs = std::max(num_fibs, mode_fibs);
              }
            }
          }
          o_fibers[p].resize(num_fibs * rank);
          total_storage += ((double) num_fibs * rank * sizeof(double)) / (1024.0*1024.0);
        } // for all partitions
        printf("ofibs storage/prtn: %f MB\n", total_storage/(double)m_num_partitions);
      }
    }
    
    void mttkrp(const int target_mode, MAT *i_factors, MAT *o_mttkrp) const {
      (*o_mttkrp).zeros();
      unsigned int rank = i_factors[target_mode].n_cols;

      // INFO << "creating output fibers for direct access during mttkrp" << std::endl;
      /*      
      if (fib_reuse <= MIN_FIBER_REUSE) {
        // Do atomic alto mttkrp
      } else {
        // Do DA-mem pull mttkrp
      }
      */

      LIT ALTO_MASKS[MAX_NUM_MODES];
      for (int n = 0; n < m_modes; ++n) {
        ALTO_MASKS[n] = mode_masks[n];
      }
      
      #pragma omp parallel for schedule(static,1) proc_bind(close)
      for (int p = 0; p < m_num_partitions; ++p) {

        //double *row = (double*)AlignedMalloc(rank * sizeof(double));
        //assert(row);
        double row[rank]; //Allocate an auto array of variable size.

        // LIT* const idx = at->idx;
        // double* const vals = at->vals;
        int const nnz_s = m_partition_ptr[p];
        int const nnz_e = m_partition_ptr[p + 1];

        for (int i = nnz_s; i < nnz_e; ++i) {
          double const val = m_alto_data[i];
          LIT const alto_idx = m_alto_indices[i];

          #pragma omp simd
          for (int r = 0; r < rank; ++r) {
            row[r] = val;
          }

          for (int m = 0; m < m_modes; ++m) {
            if (m != target_mode) { //input fibers
              int const row_id = pext(alto_idx, ALTO_MASKS[m]);
              #pragma omp simd
              for (int r = 0; r < rank; r++) {
                row[r] *= i_factors[m](row_id, r);
              }
            }
          }

          //Output fibers
          int const row_id = pext(alto_idx, ALTO_MASKS[target_mode]);
          for (int r = 0; r < rank; ++r) {
            #pragma omp atomic update
            o_mttkrp->at(r, row_id) += row[r];
          }
        } //nnzs
      } //prtns
      
      /*
      unsigned int rank = i_factors[target_mode].n_cols;

      LIT ALTO_MASKS[MAX_NUM_MODES];
      #pragma omp simd
      for (int n = 0; n < m_modes; ++n) {
        ALTO_MASKS[n] = mode_masks[n];
      }

      double row[rank]; //Allocate an auto array of variable size.

      for (int i = 0; i < m_numel; ++i) {
        LIT const alto_idx = m_alto_indices[i];
        double const val = m_alto_data[i];

        #pragma omp simd
        for (int r = 0; r < rank; ++r) {
          row[r] = val;
        }

        for (int m = 0; m < m_modes; ++m) {
          if (m != target_mode) { //input fibers
            int const row_id = pext(alto_idx, ALTO_MASKS[m]);
            #pragma omp simd
            for (int r = 0; r < rank; ++r) {
              row[r] *= i_factors[m](row_id, r);
            }
          }
        }

        //Output fibers
        int row_id = pext(alto_idx, ALTO_MASKS[target_mode]);
        #pragma omp simd
        for (int r = 0; r < rank; ++r) {
          o_mttkrp->at(r, row_id) += row[r];
        }
      } // non zeros
      */
    }
}; // class ALTOTensor
} // namespace planc

#endif // COMMON_ALTO_TENSOR_HPP_