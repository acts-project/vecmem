#include <limits>

#include "../../../utils/cuda_error_handling.hpp"
#include "vecmem/memory/cuda/notcub/allocate_host.h"
#include "vecmem/memory/cuda/notcub/getCachingDeviceAllocator.h"
#include "vecmem/memory/cuda/notcub/getCachingHostAllocator.h"

namespace {
  const size_t maxAllocationSize =
    vecmem::cuda::notcub::CachingDeviceAllocator::IntPow(vecmem::cuda::allocator::binGrowth, vecmem::cuda::allocator::maxBin);
}

namespace vecmem::cuda::notcub {
  void *allocate_host(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      VECMEM_CUDA_ERROR_CHECK(allocator::getCachingHostAllocator().HostAllocate(&ptr, nbytes, stream));
    } else {
      VECMEM_CUDA_ERROR_CHECK(cudaMallocHost(&ptr, nbytes));
    }
    return ptr;
  }

  void free_host(void *ptr) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      VECMEM_CUDA_ERROR_CHECK(allocator::getCachingHostAllocator().HostFree(ptr));
    } else {
      VECMEM_CUDA_ERROR_CHECK(cudaFreeHost(ptr));
    }
  }

}  // namespace vecmem::cuda::notcub
