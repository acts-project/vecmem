#include <cassert>
#include <limits>

#include <cuda_runtime.h>

#include "../../../utils/cuda_error_handling.hpp"

#include "vecmem/memory/cuda/notcub/ScopedSetDevice.h"
#include "vecmem/memory/cuda/notcub/allocate_device.h"
#include "vecmem/memory/cuda/notcub/getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
    vecmem::cuda::notcub::CachingDeviceAllocator::IntPow(vecmem::cuda::allocator::binGrowth, vecmem::cuda::allocator::maxBin);
}

namespace vecmem::cuda::notcub {
  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      VECMEM_CUDA_ERROR_CHECK(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(dev);
      VECMEM_CUDA_ERROR_CHECK(cudaMallocAsync(&ptr, nbytes, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      VECMEM_CUDA_ERROR_CHECK(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

  void free_device(int device, void *ptr, cudaStream_t stream) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      VECMEM_CUDA_ERROR_CHECK(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(device);
      VECMEM_CUDA_ERROR_CHECK(cudaFreeAsync(ptr, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      VECMEM_CUDA_ERROR_CHECK(cudaFree(ptr));
    }
  }

}  // namespace vecmem::cuda::notcub
