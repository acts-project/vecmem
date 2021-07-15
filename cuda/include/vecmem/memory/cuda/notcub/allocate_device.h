#ifndef HeterogeneousCore_CUDAUtilities_allocate_device_h
#define HeterogeneousCore_CUDAUtilities_allocate_device_h

#include <cuda_runtime.h>

namespace vecmem {
  namespace cuda {
    namespace notcub {
      // Allocate device memory
      void *allocate_device(int device, size_t nbytes, cudaStream_t stream);
      
      // Free device memory (to be called from unique_ptr)
      void free_device(int device, void *ptr, cudaStream_t stream);
    }  // namespace notcub
  }  // namespace cuda
} // namespace vecmem

#endif
