/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/cuda/copy_to_host.hpp"
#include "../cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace vecmem::cuda::details {

   void copy_to_host( std::size_t size, const void* devicePtr,
                      void* hostPtr ) {

      VECMEM_CUDA_ERROR_CHECK( cudaMemcpy( hostPtr, devicePtr, size,
                                           cudaMemcpyDeviceToHost ) );
   }

} // namespace vecmem::cuda::details
