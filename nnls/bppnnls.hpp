/*Copyright 2016 Ramakrishnan Kannan*/

#ifndef NNLS_BPPNNLS_HPP_
#define NNLS_BPPNNLS_HPP_

#include <assert.h>
#include "nnls.hpp"
#include "utils.hpp"
#include <set>
#include <algorithm>
#include <iomanip>
#include "SortBooleanMatrix.hpp"

// #include "cuda_utils.h"
// #include "blco.h"

// #define PROPOSED_METHOD
#define DEBUG 0

/**
 * Computes the rank-down inverse of a submatrix
 * A: matrix to rank
 * columns_to_drop
*/
MAT submatrix_inv_blocked(const MAT& A, const UVEC& columns_to_drop) {
    // Check if columns_to_drop is empty
    if (columns_to_drop.is_empty()) {
        INFO << "No columns to drop... skipping..." << "\n";
        return A;
    }

    // Perform the matrix operations
    MAT sub_A = A.cols(columns_to_drop);
    MAT inv_sub_A = arma::inv(A(columns_to_drop, columns_to_drop));
    return A - sub_A * inv_sub_A * sub_A.t();
}


template <class MATTYPE, class VECTYPE>
class BPPNNLS : public NNLS<MATTYPE, VECTYPE> {
 public:
    BPPNNLS(MATTYPE input, VECTYPE rhs, bool prodSent = false):
        NNLS<MATTYPE, VECTYPE>(input, rhs, prodSent) {
    }
    BPPNNLS(MATTYPE input, MATTYPE RHS, bool prodSent = false) :
        NNLS<MATTYPE, VECTYPE>(input, RHS, prodSent) {
    }
    int solveNNLS() {
        int rcIterations = 0;
        if (this->k == 1) {
            rcIterations = solveNNLSOneRHS();
        } else {
            // k must be greater than 1 in this case.
            // we initialized k appropriately in the
            // constructor.
            rcIterations = solveNNLSMultipleRHS();
        }
        return rcIterations;
    }
    MAT AtA_inv;

 private:
    /**
     * This implementation is based on Algorithm 1 on Page 6 of paper
     * http://www.cc.gatech.edu/~hpark/papers/SISC_082117RR_Kim_Park.pdf.
     *
     * Special case of the multi RHS solver.
     */
    int solveNNLSOneRHS() {
        // Set the RHS matrix
        this->AtB.zeros(this->n, this->k);
        this->AtB.col(0) = this->Atb;

        // Initialize the solution matrix
        this->X.zeros(this->n, this->k);
        this->X.col(0) = this->x;

        // Call matrix method
        int iter = solveNNLSMultipleRHS();

        this->x = this->X.col(0);

        return iter;
    }

    int solveNNLSMultipleRHS_int8() {
        
    }
    /**
     * This is the implementation of Algorithm 2 at Page 8 of the paper
     * http:// www.cc.gatech.edu/~hpark/papers/SISC_082117RR_Kim_Park.pdf.
     * 
     * Based on the nnlsm_blockpivot subroutine from the MATLAB code 
     * associated with the paper.
     */
    int solveNNLSMultipleRHS() {
        double wtime_matops = 0.0;
        double wtime_check_feas = 0.0;
        double wtime_update_partitions = 0.0;
        double wtime_solve_ls = 0.0;
        double wtime = 0.0;

#ifdef PROPOSED_METHOD
        MAT Q, R;
        arma::qr(Q, R, this->AtA);
        this->AtA_inv = arma::solve(R, Q.t());
#endif

        wtime = omp_get_wtime();

        UINT iter = 0;
        // UINT MAX_ITERATIONS = this->n * 5;
        UINT MAX_ITERATIONS = 1; // for profiling
        bool success = true;

        // Set the initial feasible solution
        MATTYPE Y = (this->AtA * this->X) - this->AtB;

        wtime_matops += omp_get_wtime() - wtime;

        wtime = omp_get_wtime();
        UMAT PassiveSet = (this->X > 0);

        // INFO << "this->AtB -- mttkrp" << "\n";
        // INFO << this->AtB << std::endl; // entries that alreay satisfy constraint
        // exit(0);
        int pbar = 3;
        UROWVEC P(this->k);
        P.fill(pbar);

        UROWVEC Ninf(this->k);
        Ninf.fill(this->n+1);

        UMAT NonOptSet  = (Y < 0) % (PassiveSet == 0);
        UMAT InfeaSet   = (this->X < 0) % PassiveSet;
        UROWVEC NotGood = arma::sum(NonOptSet) + arma::sum(InfeaSet);
        UROWVEC NotOptCols = (NotGood > 0);

        // INFO << PassiveSet << "\n";
        // INFO << NonOptSet << "\n";
        // INFO << InfeaSet << "\n";
        
        // INFO << NonOptSet << std::endl;
        // INFO << InfeaSet << std::endl;
        // INFO << arma::sum(NonOptSet) << std::endl;
        // INFO << NotGood << std::endl;
        // INFO << NotOptCols << std::endl;

        UWORD numNonOptCols = arma::accu(NotOptCols);
#ifdef _VERBOSE
        INFO << "Rank : " << arma::rank(this->AtA) << endl;
        INFO << "Condition : " << cond(this->AtA) << endl;
        INFO << "numNonOptCols : " << numNonOptCols;
#endif
        // Temporaries needed in loop
        UROWVEC Cols1 = NotOptCols;
        UROWVEC Cols2 = NotOptCols;
        UMAT PSetBits = NonOptSet;
        UMAT POffBits = InfeaSet;
        // UMAT NotOptMask = arma::ones<UMAT>(arma::size(NonOptSet));
        
        wtime_check_feas += omp_get_wtime() - wtime;
#if DEBUG==1
        INFO << "PassiveSet\n" << PassiveSet << "\n";
#endif
        while (numNonOptCols > 0) {
            // INFO << "PassiveSet\n" << PassiveSet << "\n";
            // printf("numNonOptCols: %d\n", numNonOptCols);
            iter++;

            if ((MAX_ITERATIONS > 0) && (iter > MAX_ITERATIONS)) {
                success = false;
                break;
            }

#if DEBUG==1
            INFO << "NotGood\n" << NotGood << "\n";
            INFO << "Ninf\n" << Ninf << "\n";
#endif

            wtime = omp_get_wtime();
            Cols1 = NotOptCols % (NotGood < Ninf);
            Cols2 = NotOptCols % (NotGood >= Ninf) % (P >= 1);
            UROWVEC Cols3Ix = arma::conv_to<UROWVEC>::from(
                        arma::find(NotOptCols % (Cols1 == 0) % (Cols2 == 0)));

            // Columns that didn't increase number of infeasible variables
            if (!Cols1.empty()) {
                // INFO << "Cols1\n" << Cols1 << "\n";
                // P(Cols1) = pbar;,Ninf(Cols1) = NotGood(Cols1);
                P(arma::find(Cols1)).fill(pbar);
                Ninf(arma::find(Cols1)) = NotGood(arma::find(Cols1));
                // INFO << "Ninf\n" << Ninf << "\n";
                // PassiveSet(NonOptSet & repmat(Cols1,n,1)) = true;
                PSetBits = NonOptSet;
                PSetBits.each_row() %= Cols1;
                PassiveSet(arma::find(PSetBits)).fill(1u);

                // PassiveSet(InfeaSet & repmat(Cols1,n,1)) = false;
                POffBits = InfeaSet;
                POffBits.each_row() %= Cols1;
                PassiveSet(arma::find(POffBits)).fill(0u);
            }

            // Columns that did increase number of infeasible variables but full
            // exchange is still allowed
            if (!Cols2.empty()) {
                // INFO << "Cols2\n" << Cols2 << "\n";

                // P(Cols2) = P(Cols2)-1;
                // INFO << "P-before\n" << P << "\n";
                P(arma::find(Cols2)) -= 1;
                // INFO << "P-after\n" << P << "\n";

                // PassiveSet(NonOptSet & repmat(Cols2,n,1)) = true;
                PSetBits = NonOptSet;
                PSetBits.each_row() %= Cols2;
                PassiveSet(arma::find(PSetBits)).fill(1u);

                // PassiveSet(InfeaSet & repmat(Cols2,n,1)) = false;
                POffBits = InfeaSet;
                POffBits.each_row() %= Cols2;
                PassiveSet(arma::find(POffBits)).fill(0u);
            }

            // Columns using backup rule
            if (!Cols3Ix.empty()) {
                // GPU offloaded version doesn't handle Cols3 -- trying to see if this happens at all
                // Remove once developement of BPP offload is complete
                // INFO << "Cols3--" << "\n";
                // exit(1);

                UROWVEC::iterator citr;
                for (citr = Cols3Ix.begin(); citr !=  Cols3Ix.end(); ++citr) {
                    UWORD colidx = *citr;
                    // find max row idx
                    UWORD rowidx = arma::max(arma::find(NonOptSet.col(colidx) + InfeaSet.col(colidx)));
                    if (PassiveSet(rowidx, colidx) > 0) { // if 1 then turn to 0
                        PassiveSet(rowidx, colidx) = 0u;
                    } else {
                        PassiveSet(rowidx, colidx) = 1u;
                    }
                }
            }

            // INFO << "X->cols(0)\n" << this->AtA << "\n";
            // INFO << "this->AtB.cols(0)\n" << this->AtB << "\n";
            UVEC NotOptColsIx = arma::find(NotOptCols);
            wtime_update_partitions += omp_get_wtime() - wtime;
            // INFO << arma::sum(PassiveSet.cols(NotOptColsIx)) << "\n";
            wtime = omp_get_wtime();

            // INFO << this->AtA << "\n";
            // INFO << arma::chol(this->AtA) << "\n";
            this->X.cols(NotOptColsIx) = solveNormalEqComb(this->AtA,
                                   this->AtB.cols(NotOptColsIx),
                                   PassiveSet.cols(NotOptColsIx));
            wtime_solve_ls+=omp_get_wtime() - wtime;

            // INFO << "X" << "\n";
            // INFO << this->X << std::endl;     

            // INFO << "MTTKRP" << "\n";
            // INFO << this->AtB << "\n";   

#if DEBUG==1
            INFO << "X" << "\n";
            INFO << this->X << std::endl;        
#endif
            wtime = omp_get_wtime();

            // INFO << "AtA * X" << "\n";
            // INFO << this->AtA * this->X << std::endl;        
            
            Y.cols(NotOptColsIx) = (this->AtA * this->X.cols(NotOptColsIx))
                             - this->AtB.cols(NotOptColsIx);

            wtime_matops += omp_get_wtime() - wtime;

            // Y = (this->AtA * this->X) - this->AtB;
            // X(abs(X)<1e-12) = 0;

            // INFO << "gram * X\n" << this->AtA << "\n";
            fixAbsNumericalError<MATTYPE>(&this->X, EPSILON_1EMINUS12, 0.0);
            // Y(abs(Y)<1e-12) = 0;

            fixAbsNumericalError<MATTYPE>(&Y, EPSILON_1EMINUS12, 0.0);
            // INFO << "PassiveSet\n" << PassiveSet << "\n";
#if DEBUG==1
            // INFO << "X after fix\n" << this->X << "\n";
            INFO << "Y after fix\n" << Y << "\n";
#endif
            // NotOptMask = repmat(NotOptCols,n,1);
            // NotOptMask.ones();
            // NotOptMask.each_row() %= NotOptCols;

            wtime = omp_get_wtime();
            NonOptSet  = (Y < 0) % (PassiveSet == 0);
            InfeaSet   = (this->X < 0) % PassiveSet;
            NotGood = arma::sum(NonOptSet) + arma::sum(InfeaSet);
            NotOptCols = (NotGood > 0);
            numNonOptCols = arma::accu(NotOptCols);

            wtime_check_feas += omp_get_wtime() - wtime;

            // INFO << "numNonOptCols-post\n" << numNonOptCols << "\n";
#if DEBUG==1
            INFO << "NonOptSet\n" << NonOptSet << "\n";
            INFO << "InfeaSet\n" << InfeaSet << "\n";
            INFO << "NotGood-post\n" << NotGood << "\n";
#endif

        }

        if (!success) {
            ERR << "BPP failed" << std::endl;
            // exit(EXIT_FAILURE);
        }

        printf("matops\t%f\nfeas_check\t%f\nupdate_part\t%f\nbatch_solve\t%f\n", 1e3*wtime_matops, 1e3*wtime_check_feas, 1e3*wtime_update_partitions, 1e3*wtime_solve_ls);
        return iter;
    }

    /**
     * This function to support the step 10 of the algorithm 2.
     * This is implementation of the paper
     * Fast algorithm for the solution of large-scale non-negativity-constrained least squares problems
     * M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450
     * Motivated out of implementation from Jingu's solveNormalEqComb.m
     * 
     * @param[in] LHS of the system of size \f$n \times n\f$
     * @param[in] RHS of the system of size \f$n \times nrhs\f$
     * @param[in] Binary matrix of size \f$n \times nrhs\f$ representing the Passive Set
     */
    MATTYPE solveNormalEqComb(MATTYPE AtA, MATTYPE AtB, UMAT PassSet) {
        MATTYPE Z;
        UVEC anyZeros = arma::find(PassSet == 0);
        // INFO << "PassSet\n" << PassSet << std::endl;
        if (anyZeros.empty()) { // PassSet is all 1's
            // printf("everything in passive set\n");
            // Everything is the in the passive set.
            Z = arma::solve(AtA, AtB, arma::solve_opts::likely_sympd);
        } else {
            // printf("everything not in passive set\n");
            UVEC Pv = arma::find(PassSet != 0);

            Z.resize(AtB.n_rows, AtB.n_cols);
            Z.zeros();

            UINT k1 = PassSet.n_cols; 
            if (k1 == 1) {
                // printf("k1 == 1\n");

                // Single column to solve for.
                Z(Pv) = arma::solve(AtA(Pv, Pv), AtB(Pv),
                                arma::solve_opts::likely_sympd);
            } else {
                // printf("k1 != 1\n");

                // we have to group passive set columns that are same.
                // find the correlation matrix of passive set matrix.

                double _wtime = omp_get_wtime();

                std::vector<UWORD> sortedIdx, beginIdx;
                computeCorrelationScore(PassSet, sortedIdx, beginIdx);

                // printf("grouping time: %f\n", omp_get_wtime() - _wtime);
                int num_chol_saved = 0;
/*
                std::cout << "sortedIdx: ";
                for (const auto& element : sortedIdx) {
                    std::cout << element << " ";
                }
                std::cout << std::endl;

                // Printing beginIdx
                std::cout << "beginIdx: ";
                for (const auto& element : beginIdx) {
                    std::cout << element << " ";
                }
                std::cout << std::endl;

*/

                 // printf("beginIdx size: %d\n", beginIdx.size());
                // Go through the groups one at a time
                double wtime;
                double wtime_solve = 0.0;

                double __wtime = 0.0;
                double ___wtime = 0.0;

                for (UINT i = 1; i < beginIdx.size(); i++) {
                    UWORD sortedBeginIdx = beginIdx[i - 1];
                    UWORD sortedEndIdx = beginIdx[i];
                    if (sortedEndIdx - sortedBeginIdx > 1) {
                        num_chol_saved += sortedEndIdx - sortedBeginIdx - 1;
                        // printf("# of columns with same pattern: %d\n", sortedEndIdx - sortedBeginIdx + 1);
                    }

                    // Create submatrices of indices for solve.
                    UVEC samePassiveSetCols(std::vector<UWORD>
                                            (sortedIdx.begin() + sortedBeginIdx,
                                             sortedIdx.begin() + sortedEndIdx));
                    UVEC currentPassiveSet = arma::find(
                            PassSet.col(sortedIdx[sortedBeginIdx]) == 1);
#if 0
                    INFO << "samePassiveSetCols::" << std::endl
                         <<  samePassiveSetCols << std::endl;
                    INFO << "currPassiveSet::" << std::endl
                         << currentPassiveSet << std::endl;
                    INFO << "AtA::" << std::endl
                         << AtA(currentPassiveSet, currentPassiveSet)
                         << std::endl;
                    INFO << "AtB::" << std::endl
                         << AtB(currentPassiveSet, samePassiveSetCols)
                         << std::endl;
#endif // verbose logging


#ifdef PROPOSED_METHOD
                    // compute explicit inverse of entire gram matrix
                    // itentify currentPassiveSet
                    _wtime = omp_get_wtime();
                    UVEC columns_to_drop(this->n - currentPassiveSet.size());

                    int cnt = 0;
                    for (int i = 0; i < this->n; ++i) {
                        if (!arma::any(i == currentPassiveSet)) {
                            columns_to_drop[cnt] = i;
                            cnt++;
                        }
                    }
                    
                    // INFO << columns_to_drop << "\n";
                    // compute rank-down
                    // do matrix mult
                    MAT AtA_rankdown = submatrix_inv_blocked(this->AtA_inv, columns_to_drop);

                    __wtime += omp_get_wtime() - _wtime;
                    _wtime = omp_get_wtime();
                    // AtB has to be a copied one with 0-ed out
                    // INFO << AtA_rankdown << "\n";

                    Z(currentPassiveSet, samePassiveSetCols) = AtA_rankdown(currentPassiveSet, currentPassiveSet) * AtB(currentPassiveSet, samePassiveSetCols);
                    ___wtime += omp_get_wtime() - _wtime;
#else
                    Z(currentPassiveSet, samePassiveSetCols) = arma::solve(
                            AtA(currentPassiveSet, currentPassiveSet),
                            AtB(currentPassiveSet, samePassiveSetCols),
                            arma::solve_opts::likely_sympd);
#endif

                    // Lets change this solve to compare the cost of decomposition
                    // And actual solve
                    // wtime = omp_get_wtime();
                    // MAT L = arma::chol(AtA(currentPassiveSet, currentPassiveSet), "lower");
                    // wtime_decomp += omp_get_wtime() - wtime;

                    // wtime = omp_get_wtime();
                    // MAT temp = arma::solve(arma::trimatl(L), AtB(currentPassiveSet, samePassiveSetCols));
                    // Z(currentPassiveSet, samePassiveSetCols) = arma::solve(arma::trimatu(L.t()), temp);
                    // wtime_solve += omp_get_wtime() - wtime;

                }
                // printf("downdate,matmul,%f,%f\n", __wtime, ___wtime);
                // printf("beginIdx.size(): %d\n", beginIdx.size());
                // printf("%d\t%d\tGE\t%f\n", num_chol_saved, PassSet.n_cols, (double)(num_chol_saved) / (double)PassSet.n_cols);
                // printf("decomp\t%f\tsolve\t%f\n", wtime_decomp / (wtime_solve + wtime_decomp), wtime_solve / (wtime_solve + wtime_decomp));
// #endif // IS_GPU_OFFLOAD==1

// #ifdef IS_GPU_OFFLOAD
                // for (int i = 0; i < beginIdx.size(); ++i) {
                //     cudaStreamDestroy(streams[i]);
                // }
// #endif
            }
        }
#ifdef _VERBOSE
        INFO << "Returning mat Z:" << std::endl << Z;
#endif
        return Z;
    }

   /**
    * Passset is a binary matrix where every column represents
    * one datapoint. The objective is to returns a low triangular
    * correlation matrix with 1 if the strings are equal. Zero otherwise
    * 
    * @param[in] The binary matrix being grouped
    * @param[in] Reference to the array containing lexicographically sorted
    *            columns of the binary matrix
    * @param[in] Running indices of the grouped columns in the sorted index
    *            array
    */
    void computeCorrelationScore(UMAT &PassSet, std::vector<UWORD> &sortedIdx,
                                 std::vector<UWORD> &beginIndex) {
        SortBooleanMatrix<UMAT> sbm(PassSet);
        sortedIdx = sbm.sortIndex();
        BooleanArrayComparator<UMAT> bac(PassSet);
        uint beginIdx = 0;
        beginIndex.clear();
        beginIndex.push_back(beginIdx);
        for (uint i = 0; i < sortedIdx.size(); i++) {
            if (i == sortedIdx.size() - 1 ||
                    bac(sortedIdx[i], sortedIdx[i + 1]) == true) {
                beginIdx = i + 1;
                beginIndex.push_back(beginIdx);
            }
        }
    }
};

#endif  // NNLS_BPPNNLS_HPP_