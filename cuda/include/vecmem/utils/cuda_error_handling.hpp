/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include <cuda_runtime_api.h>

// Detray Data Model include(s).
#include "vecmem/utils/types.hpp"

/// Helper macro used for checking @c cudaError_t type return values.
#   define VECMEM_CUDA_ERROR_CHECK( EXP )                                      \
   do {                                                                        \
      cudaError_t errorCode = EXP;                                             \
      if( errorCode != cudaSuccess ) {                                         \
        vecmem::cuda::details::throw_error( errorCode, #EXP, __FILE__,         \
                                            __LINE__ );                        \
      }                                                                        \
   } while( false )

/// Helper macro used for running a CUDA function when not caring about its results
#ifdef __CUDACC__
#   define VECMEM_CUDA_ERROR_IGNORE( EXP )                                     \
   do {                                                                        \
      EXP;                                                                     \
   } while( false )
#else
#   define VECMEM_CUDA_ERROR_IGNORE( EXP ) do {} while( false )
#endif // __CUDACC__

namespace vecmem { namespace cuda { namespace details {

   /// Function used to print and throw a user-readable error if something breaks
   void throw_error( cudaError_t errorCode, const char* expression,
                     const char* file, int line );

} } } // namespace vecmem::cuda::details
