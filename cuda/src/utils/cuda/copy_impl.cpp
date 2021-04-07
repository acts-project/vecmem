/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/debug.hpp"
#include "vecmem/utils/cuda/impl/copy_impl.hpp"
#include "../cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <string>

namespace vecmem::cuda::details {

   /// The number of "copy types"
   static constexpr std::size_t n_copy_types = 5;

   /// Helper array for translating between the vecmem and CUDA copy type
   /// definitions
   static constexpr cudaMemcpyKind copy_type_translator[ n_copy_types ] = {
      cudaMemcpyHostToDevice,
      cudaMemcpyDeviceToHost,
      cudaMemcpyHostToHost,
      cudaMemcpyDeviceToDevice,
      cudaMemcpyDefault
   };

   /// Helper array for providing a printable name for the copy type definitions
   static const std::string copy_type_printer[ n_copy_types ] = {
      "host to device",
      "device to host",
      "host to host",
      "device to device",
      "unknown"
   };

   void copy_raw( std::size_t size, const void* from, void* to,
                  memory_copy_type type ) {

      // Check if anything needs to be done.
      if( size == 0 ) {
         VECMEM_DEBUG_MSG( 1, "Skipping unnecessary memory copy" );
         return;
      }

      // Some sanity checks.
      assert( from != nullptr );
      assert( to != nullptr );
      assert( static_cast< int >( type ) >= 0 );
      assert( static_cast< int >( type ) < static_cast< int >( n_copy_types ) );

      // Perform the copy.
      VECMEM_CUDA_ERROR_CHECK( cudaMemcpy( to, from, size,
                                           copy_type_translator[ type ] ) );

      // Let the user know what happened.
      VECMEM_DEBUG_MSG( 4, "Performed %s memory copy of %lu bytes from %p to "
                        "%p", copy_type_printer[ type ].c_str(), size, from,
                        to );
   }

} // namespace vecmem::cuda::details
