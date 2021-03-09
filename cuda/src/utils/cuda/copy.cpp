/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/cuda/copy.hpp"
#include "../cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace vecmem::cuda::details {

   void copy_to_device( std::size_t size, const void* hostPtr,
                        void* devicePtr ) {

      VECMEM_CUDA_ERROR_CHECK( cudaMemcpy( devicePtr, hostPtr, size,
                                           cudaMemcpyHostToDevice ) );
   }

   void copy_to_host( std::size_t size, const void* devicePtr,
                      void* hostPtr ) {

      VECMEM_CUDA_ERROR_CHECK( cudaMemcpy( hostPtr, devicePtr, size,
                                           cudaMemcpyDeviceToHost ) );
   }

   void copy( std::size_t size, const void* from, void* to ) {

      VECMEM_CUDA_ERROR_CHECK( cudaMemcpy( to, from, size,
                                           cudaMemcpyDefault ) );
   }

} // namespace vecmem::cuda::details
