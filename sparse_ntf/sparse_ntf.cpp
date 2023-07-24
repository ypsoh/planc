#include <iostream>
#include "common/utils.h"
#include "common/ncpfactors.hpp"
#include "common/ntf_utils.hpp"
#include "common/parsecommandline.hpp"
// #include "common/tensor.hpp"
#include "common/sparse_tensor.hpp"
#include "ntf/ntfanlsbpp.hpp"
#include "ntf/ntfaoadmm.hpp"
#include "ntf/ntfhals.hpp"
#include "ntf/ntfmu.hpp"
#include "ntf/ntfnes.hpp"

namespace planc {

class SparseNTFDriver {
    public:
        template <class NTFType>
        void callNTF(planc::ParseCommandLine pc) {
            std::string filename = pc.input_file_name();

            if (filename.empty()) {
                std::cout << "Input filename required for SparseNTF operations..." << std::endl;
                exit(1);
            }
            SparseTensor my_tensor(filename);
            my_tensor.map_to_compact_indices();
            for (int m = 0; m < my_tensor.modes(); ++m) {
                std::cout << "number of dims for mode: " << m << " is: " << my_tensor.dimensions(m) << std::endl;
            }

            // int test_modes = pc.num_modes();
            // UVEC dimensions(test_modes);

            // Load tensor
            // get the number of modes e.g. 4
            // gets the dimensions e.g. 23 423 234
            // get the SparseTensor

            std::cout << "Input filename = " << filename << std::endl;


        }
};
}

int main(int argc, char* argv[]) {
    planc::ParseCommandLine pc(argc, argv);
    pc.parseplancopts();
    planc::SparseNTFDriver sntfd;
    switch (pc.lucalgo())
    {
    case MU:
        sntfd.callNTF<planc::NTFMU>(pc);
        break;
    
    default:
        ERR << "Wrong algorithm choice. Quitting.." << pc.lucalgo() << std::endl;
    }
}