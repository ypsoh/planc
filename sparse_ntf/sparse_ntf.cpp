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
    // template <class NTFType>
    template <template<class T> class NTFType, class T>
    void callNTF(planc::ParseCommandLine pc) {
      std::string filename = pc.input_file_name();
      if (filename.empty()) {
        std::cout << "Input filename required for SparseNTF operations..." << std::endl;
        exit(1);
      }

      T my_tensor(filename);
      my_tensor.print();

      NTFType<T> ntfsolver(my_tensor, pc.lowrankk(), pc.lucalgo());
      // Setting flags. does it need to be here?
      ntfsolver.num_it(pc.iterations());
      ntfsolver.compute_error(pc.compute_error());
      ntfsolver.computeSparseNTF();
    }
    SparseNTFDriver() {}
};
}

int main(int argc, char* argv[]) {
  planc::ParseCommandLine pc(argc, argv);
  pc.parseplancopts();
  planc::SparseNTFDriver sntfd;
  switch (pc.lucalgo())
  {
  case MU:
    sntfd.callNTF<planc::NTFMU, planc::SparseTensor>(pc);
    break;

  default:
    ERR << "Wrong algorithm choice. Quitting.." << pc.lucalgo() << std::endl;
  }
}