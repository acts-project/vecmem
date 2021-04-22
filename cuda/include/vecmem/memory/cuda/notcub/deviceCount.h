#ifndef HeterogenousCore_CUDAUtilities_deviceCount_h
#define HeterogenousCore_CUDAUtilities_deviceCount_h

#include "../../../../../src/utils/cuda_error_handling.hpp"

#include <cuda_runtime.h>

namespace vecmem {
  namespace cuda {
    namespace notcub {
      inline int deviceCount() {
        int ndevices;
        cudaCheck(cudaGetDeviceCount(&ndevices));
        return ndevices;
      }
    } // namespace notcub
  }  // namespace cuda
}  // namespace vecmem

#endif
