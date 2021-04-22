#ifndef HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h
#define HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h

#include "../../../../../src/utils/cuda_error_handling.hpp"

#include <cuda_runtime.h>

namespace vecmem {
  namespace cuda {
    namespace notcub {
      class ScopedSetDevice {
      public:
        explicit ScopedSetDevice(int newDevice) {
          VECMEM_CUDA_ERROR_CHECK(cudaGetDevice(&prevDevice_));
          VECMEM_CUDA_ERROR_CHECK(cudaSetDevice(newDevice));
        }
        
        ~ScopedSetDevice() {
          // Intentionally don't check the return value to avoid
          // exceptions to be thrown. If this call fails, the process is
          // doomed anyway.
          cudaSetDevice(prevDevice_);
        }
        
      private:
        int prevDevice_;
      };
    } // namespace notcub
  }  // namespace cuda
}  // namespace vecmem

#endif
