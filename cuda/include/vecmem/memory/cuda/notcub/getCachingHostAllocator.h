#ifndef HeterogeneousCore_CUDACore_src_getCachingHostAllocator
#define HeterogeneousCore_CUDACore_src_getCachingHostAllocator

#include <iomanip>
#include <iostream>

#include "../../../../../src/utils/cuda_error_handling.hpp"
#include "CachingHostAllocator.h"
#include "getCachingDeviceAllocator.h"

namespace vecmem::cuda::allocator {
  inline notcub::CachingHostAllocator& getCachingHostAllocator() {
    if (debug) {
      std::cout << "cub::CachingHostAllocator settings\n"
                << "  bin growth " << binGrowth << "\n"
                << "  min bin    " << minBin << "\n"
                << "  max bin    " << maxBin << "\n"
                << "  resulting bins:\n";
      for (auto bin = minBin; bin <= maxBin; ++bin) {
        auto binSize = notcub::CachingDeviceAllocator::IntPow(binGrowth, bin);
        if (binSize >= (1 << 30) and binSize % (1 << 30) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 30) << " GB\n";
        } else if (binSize >= (1 << 20) and binSize % (1 << 20) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 20) << " MB\n";
        } else if (binSize >= (1 << 10) and binSize % (1 << 10) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 10) << " kB\n";
        } else {
          std::cout << "    " << std::setw(9) << binSize << " B\n";
        }
      }
      std::cout << "  maximum amount of cached memory: " << (minCachedBytes() >> 20) << " MB\n";
    }

    // the public interface is thread safe
    static notcub::CachingHostAllocator allocator{binGrowth,
                                                  minBin,
                                                  maxBin,
                                                  minCachedBytes(),
                                                  false,  // do not skip cleanup
                                                  debug};
    return allocator;
  }
}  // namespace vecmem::cuda::allocator

#endif
