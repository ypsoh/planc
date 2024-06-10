/* Copyright 2016 Ramakrishnan Kannan */

#ifndef NNLS_SORTBOOLEANMATRIX_HPP_
#define NNLS_SORTBOOLEANMATRIX_HPP_
#include <armadillo>
#include <vector>
#include <algorithm>

template <class T>
class BooleanArrayComparator {
    const T &X;
 public:
    explicit BooleanArrayComparator(const T &input): X(input) {
    }
    /*
    * if idxi < idxj return true;
    * if idxi >= idxj return false;
    */
    // bool operator() (UWORD idxi, UWORD idxj) {
    //     for (uint i = 0; i < X.n_rows; i++) {
    //         if (this->X(i, idxi) < this->X(i, idxj))
    //             return true;
    //         else if (this->X(i, idxi) > this->X(i, idxj))
    //             return false;
    //     }
    //     return false;
    // }

    // first boolean (a>b) checks the count of 1s
    // second boolean term checks the size of idxi and idxj when a and b are the same
    /**
     * scenario 1 return false
     * idxi 1 1 0 0 a=2, is_a_larger=true 
     * idxj 0 1 1 1 b=3
     * 
     * scenario 2 return true
     * idxi 1 1 0 0 a=2 is_a_larger=false
     * idxj 1 1 0 0 b=2
     * 
     * scenario 3 return true
     * idxi 1 1 0 0 a=2 is_a_larger=true
     * idxj 0 0 1 1 b=2
     * 
     * scenario 4 return false
     * idxi 1 1 0 0 a=2 is_a_larger_false
     * idxj 1 1 0 1 b=3
     * 
     * scenario 5 return true
     * idxi 1 1 1 0 a = 3 is_a_larger=true
     * idxj 1 1 0 0 b = 2
    */
    // bool operator() (UWORD idxi, UWORD idxj) {
    //     UWORD a = 0;
    //     UWORD b = 0;
    //     bool is_a_bigger = false;
    //     bool is_set = false;
        
    //     for (uint i = 0; i < X.n_rows; i++) {
    //         if (this->X(i, idxi) == 1) a++;
    //         if (this->X(i, idxj) == 1) b++;
    //         if (!is_set && (this->X(i, idxi) > this->X(i, idxj)))
    //             is_a_bigger = true;
    //     }
    //     printf("a, b, is_a_bigger, %d, %d, %d\n",a,b,is_a_bigger);
    //     return (a == b) ? is_a_bigger : (a > b);
    // }

    bool operator() (UWORD idxi, UWORD idxj) {
    UWORD a = 0;
    UWORD b = 0;
    bool is_a_bigger = false;
    bool is_set = false;
    
    for (uint i = 0; i < X.n_rows; i++) {
        if (this->X(i, idxi) == 1) a++;
        if (this->X(i, idxj) == 1) b++;
        if (!is_set) {
            if (this->X(i, idxi) > this->X(i, idxj)) {
                is_a_bigger = true;
                is_set = true;
            } else if (this->X(i, idxi) < this->X(i, idxj)) {
                is_a_bigger = false;
                is_set = true;
            }
        }
    }
    // printf("a, b, is_a_bigger, %d, %d, %d\n",a,b,is_a_bigger);
    return (a == b) ? is_a_bigger : (a > b);
    }
};

template <class T>
class SortBooleanMatrix {
    const T &X;
    std::vector<UWORD> idxs;
 public:
    explicit SortBooleanMatrix(const T &input) : X(input), idxs(X.n_cols) {
        for (uint i = 0; i < X.n_cols; i++) {
            idxs[i] = i;
        }
    }
    std::vector<UWORD> sortIndex() {
        std::sort(this->idxs.begin(), this->idxs.end(),
             BooleanArrayComparator<T>(this->X));
        return this->idxs;
    }
};
#endif  // NNLS_SORTBOOLEANMATRIX_HPP_
