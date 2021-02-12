/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Detray Data Model include(s).
#include "vecmem/utils/types.hpp"

/// Helper macro used for checking @c hipError_t type return values.
#ifdef __HIP__
#   define VECMEM_HIP_ERROR_CHECK( EXP )                                       \
   do {                                                                        \
      hipError_t errorCode = EXP;                                              \
      if( errorCode != hipSuccess ) {                                          \
        vecmem::hip::details::throw_error( errorCode, #EXP, __FILE__,          \
                                            __LINE__ );                        \
      }                                                                        \
   } while( false )
#else
#   define VECMEM_HIP_ERROR_CHECK( EXP ) do {} while( false )
#endif // __CUDACC__

/// Helper macro used for running a HIP function when not caring about its results
#ifdef __HIP__
#   define VECMEM_HIP_ERROR_IGNORE( EXP )                                      \
   do {                                                                        \
      EXP;                                                                     \
   } while( false )
#else
#   define VECMEM_HIP_ERROR_IGNORE( EXP ) do {} while( false )
#endif // __CUDACC__

namespace vecmem { namespace hip { namespace details {

   /// Function used to print and throw a user-readable error if something breaks
   void throw_error( hipError_t errorCode, const char* expression,
                     const char* file, int line );

} } } // namespace vecmem::hip::details
