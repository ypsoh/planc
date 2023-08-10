#ifndef COMMON_BLCO_TENSOR_HPP_
#define COMMON_BLCO_TENSOR_HPP_

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
#include "common/alto_tensor.hpp"
#include "blco.h"
#include "cassert"

// #include "include/blcotensor_gpu"
#define _IType unsigned long long

namespace planc {
  // forward declaration
  // template <typename LIT> 
  // class BLCOTensor;
  
  // struct BLCOBlock;

  // A single block in the BLCO format. Note we force 64 bit as the LIT type
  // template <typename LIT>
  
  template <typename LIT>
  class BLCOTensor : public planc::ALTOTensor<LIT> {
    public:
      int m_num_blocks = 0;
      int m_max_block_size = 16777216;

      // for BLCO tensors
      unsigned long long * m_blco_indices;
      double* m_blco_values;

      int * m_blco_mode_pos;
      unsigned long long * m_blco_mode_mask;

      BLCOBlock ** m_blocks = nullptr;
      int m_block_count = 0;

      BLCOTensor(std::string filename) : ALTOTensor<LIT>(filename) {
        double wtime_s, wtime;

        // Pin memory on host
        this->m_blco_indices = (unsigned long long *)malloc(sizeof(unsigned long long) * this->m_numel);
        this->m_blco_values = (double *) malloc(sizeof(double) * this->m_numel);

        this->m_blco_mode_pos = (int *)malloc(sizeof(int) * MAX_NUM_MODES);
        this->m_blco_mode_mask = (unsigned long long *)malloc(sizeof(unsigned long long) * MAX_NUM_MODES);
        
        // change to cuda code
        // check_cuda(
        //   cudaMallocHost((void**)&this->m_blco_indices, sizeof(unsigned long long) * this->m_numel), "cudaMallocHost coords"); // Pinned mem
        // check_cuda(
        //   cudaMallocHost((void**)&this->m_blco_values, sizeof(double) * this->m_numel), "cudaMallocHost values");

        wtime_s = omp_get_wtime();

        int truncated_bitcounts[MAX_NUM_MODES];
        // if alto mode mask is 0100101 for a certain mode
        // truncated_bitcounts stores the number of 1s, so 3 in this case
        // if you sum it all for all modes
        // it basically represents the total number of bits used to encode ALTO tensor
        for (int m = 0; m < this->m_modes; ++m) {
          unsigned long long mask = lhalf(this->mode_masks[m]);
          truncated_bitcounts[m] = popcount(mask);
        }
        
        // construct mask and pos for BLCO
        // untangling the bit masks from alto and using consecutive bit masks for each mode
        for (int m = 0; m < this->m_modes; ++m) {
          this->m_blco_mode_mask[m] = ((unsigned long long) 1 << truncated_bitcounts[m]) - 1;
          this->m_blco_mode_pos[m] = (m == 0) ? 0 : this->m_blco_mode_pos[m-1] + truncated_bitcounts[m-1];
        }

        for (int n = 0; n < this->m_modes; ++n) {
          printf("BLCO_MASKS[%d] = 0x%llx\n", n, this->m_blco_mode_mask[n]);
        }

        // Relinearize and copy into BLCO
        #pragma omp parallel for schedule(static)
        for (unsigned long long i = 0; i < this->m_numel; i++) {
          LIT index = this->m_alto_indices[i];
          unsigned long long blco_idx = 0;
          for (int n = 0; n < this->m_modes; ++n) {
            // recover original index, re encode to linearized format
            unsigned long long mode_idx = (unsigned long long) pext(index, this->mode_masks[n]) & this->m_blco_mode_mask[n];
            blco_idx |= (mode_idx << this->m_blco_mode_pos[n]);
          }
          this->m_blco_indices[i] = blco_idx;
          this->m_blco_values[i] = this->m_alto_data[i];
        }

        wtime = omp_get_wtime() - wtime_s;
        printf("BLCO: Relinearize time = %f (s)\n", wtime);

        wtime_s = omp_get_wtime();
        // determine possible number of blocks
        int block_count = 0;
        for (int i = 0; i < this->m_modes; i++) {
          // each dimension -- count the leading zeros
          // the smaller the dimension, the larger the clz count would be
          // the larger the dimension, the smaller the clz would be
          // substract it from 64, the smaller dimensions would have smaller block_count
          // the larger dimension would have larger block count. If dimension has zero leading zero
          // then block count would be 64
          block_count += (sizeof(unsigned long long) * 8) - clz(this->m_dimensions[i] - 1);
        }
        // XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX // 128 bit
        // XXXX XXXX XXXX XXXX OOOO OOOO OOOO OOOO // front 64 bits are used for BLCO
        // need to count how much bits of "X" we need to use for block indexing
        // for all modes, 64 - clz(dim_size), basically computes how many Os we need to represent
        // a certain mode, after the sum for all modes, substract it from 64, 
        // which basically tells us how many X bits we need for block indexing
        block_count = block_count - sizeof(unsigned long long) * 8; // possible negative at this point
        block_count = std::max(0, block_count);
        // it can be 1 or 2^block_count
        block_count = 1 << block_count;

        // block_histogram counts the number of nonzeros each block has
        // note that each block uses its key (uhalf(alto_index) as its index.
        unsigned long long* block_histogram = new unsigned long long[block_count];
        unsigned long long* block_prefix_sum = new unsigned long long[block_count + 1];
        #pragma omp parallel for
        for (unsigned long long i = 0; i < block_count; i++) block_histogram[i] = 0;

        wtime = omp_get_wtime() - wtime_s;
        printf("BLCO: Setup time = %f (s), BLOCK COUNT: %d\n", wtime, block_count);

        // Construct block histogram. OpenMP 4.5+ required
        wtime_s = omp_get_wtime();
        if (block_count > 1) {
          #pragma omp parallel for reduction(+:block_histogram[:block_count])
          for (unsigned long long i = 0; i < this->m_numel; i++) {
            // Get upper half of linearized index and use that as block_id
            unsigned long long block_id = uhalf(this->m_alto_indices[i]); 
            block_histogram[block_id] += 1;
          }
        } else {
          // one block has all non-zeros
          block_histogram[0] = this->m_numel;
        }
        wtime = omp_get_wtime() - wtime_s;
        printf("BLCO: Histogram time = %f (s)\n", wtime);
        
        for (int bidx = 0; bidx < block_count; ++bidx) {
          printf("block %d: %d\n", bidx, block_histogram[bidx]);
        }
        // we may have to split the blocks further if it contains too many nnzs
        wtime_s = omp_get_wtime();
        unsigned long long total_blocks_split = 0;
        block_prefix_sum[0] = 0;
        if (this->m_max_block_size <= 0) this->m_max_block_size = this->m_numel;
        for (unsigned long long i = 0; i < block_count; i++) {
          // block_histogram[i] has the nnz for block index i
          // total_blocks_split tracks how many blocks we need to split
          // if there is too many nnz in a single block i, 
          // or in some cases some blocks might have 0 nnzs
          total_blocks_split += (block_histogram[i] + this->m_max_block_size - 1) / this->m_max_block_size;
          // block_prefix_sum accumulates the nnz for each blocks
          block_prefix_sum[i + 1] = block_prefix_sum[i] + block_histogram[i];
        }

        // Construct split blocks
        // "block_count" is the number of blocks purely based on exceeding bits > 64
        // "total_blocks_split" is "block_count" + a where a is determined based on how many
        // blocks didn't "fit" based on criteria, if no block exceeds limit, block_count == total_blocks_split
        wtime_s = omp_get_wtime();
        // BLCOBlock** blocks = new BLCOBlock*[total_blocks_split];

        this->m_blocks = new BLCOBlock*[total_blocks_split];
        this->m_block_count = total_blocks_split;
        // this->block_count = total_blocks_split;

        int curr_blocks = 0;
        for (int block = 0; block < block_count; block++) {
          int start = block_prefix_sum[block];
          int end = block_prefix_sum[block + 1];
          int nnz = end - start; // the nnz each block has
          if (nnz > 0) {
            // Generate block coordinates, tricky because we can only use "block"
            unsigned long long block_coords[MAX_NUM_MODES];
            for (int i = 0; i < this->m_modes; i++) {
              // YONGSEOK: note this->mode_masks is from ALTOTensor,, 
              // block_coords is required to reverse back to the original ALTO idnex
              unsigned long long mode_mask = uhalf(this->mode_masks[i]);
              // from ALTO perspective, mode_idx will contain the bit mask information 
              // if block index is 101 
              // and mode_mask for uhalf for a mode is 011, 
              // alto needs to know that 001 to get original coordinate
              unsigned long long mode_idx = pext(block, mode_mask);
              mode_idx <<= truncated_bitcounts[i]; // shift it back up to block idx space
              block_coords[i] = mode_idx;
            }

            // Generate split block, indices are just offset pointers into main array
            for (int stride = 0; stride < nnz; stride += this->m_max_block_size) {
              int nnz_split_block = std::min(this->m_max_block_size, nnz - stride);
              BLCOBlock* blk = generate_block_host(this->m_modes, nnz_split_block);
              blk->idx = this->m_blco_indices + start + stride;
              blk->vals = this->m_blco_values + start + stride;
              this->m_blocks[curr_blocks] = blk;
              curr_blocks++;
              for (int i = 0; i < this->m_modes; i++) blk->block_coords[i] = block_coords[i];
            }
          }
        }

        wtime = omp_get_wtime() - wtime_s;
        printf("BLCO: Blocking time = %f (s)\n", wtime);
        printf("Num. blocks: %d\tTotal blocks: %d\n\n", block_count, total_blocks_split);

        // print block stuff
        for (int block_idx = 0; block_idx < total_blocks_split; ++block_idx) {
          BLCOBlock * b = this->m_blocks[block_idx];
          printf("bidxDimensions: %d, nnz: %d\n", block_idx, b->m_numel);
          for (int m = 0; m < b->m_modes; ++m) {
            printf("block coord for mode: %d\t 0x%llx\n", m, b->block_coords[m]);
          }
        }
        // free ALTO related stuff
        //  -- not ideal and kind of hacky since we're using std::vectors
        this->m_alto_data.clear(); // doesn't guarantee
        this->m_alto_data.shrink_to_fit(); // C++11
        this->m_alto_indices.clear();
        this->m_alto_indices.shrink_to_fit();
        this->m_partition_intervals.clear();
        this->m_partition_intervals.shrink_to_fit();
        this->m_partition_ptr.clear();
        this->m_partition_ptr.shrink_to_fit();
      }

      void mttkrp(const int target_mode, MAT *i_factors, MAT *o_mttkrp) const {
        INFO << "Copying BLCO tensor to GPU device..." << std::endl;
        BLCOTensorGPU * bt = this->send_blcotensor_to_gpu();


        MAT_GPU * o_mttkrp_gpu = send_mat_to_gpu(o_mttkrp);
        MAT_GPU ** i_factors_gpu = send_mats_to_gpu(i_factors, this->m_modes);
        INFO << "Copied input factors and output mttkrp to GPU device..." << std::endl;

        // unsigned int rank = i_factors[i_n].ncols;
        // dimensions
        // need to copy num_modes x (dims[m] * rank ) x sizeof(double)


        // this->send_o_factors_to_gpu();
        // allocate_gpu_mem();

        // send_factors_to_gpu();
        // now start mttkrp
        _hello();
        exit(1);
      }

      BLCOTensorGPU * send_blcotensor_to_gpu() const {
        bool stream_data = false; // change afterwards
        bool do_batching = false; // change afterwards

        _IType num_streams = 8;
        _IType max_block_size = this->m_max_block_size;
        if (!stream_data) {
          num_streams = this->m_block_count;
          max_block_size = 0;
        }

        BLCOTensorGPU * blco_dev = copy_blcotensor_to_device(
          max_block_size,
          this->m_modes,
          (unsigned long long *)(this->m_dimensions.memptr()),
          this->m_numel,
          this->m_blco_mode_mask,
          this->m_blco_mode_pos,
          this->m_block_count,
          this->m_blocks
        );
        INFO << "send_blco_tensor_to_gpu() generated blco tensor on device" << std::endl;
        return blco_dev;
      }

  };

}

#endif