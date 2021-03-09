/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/array.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"
#include "test_cuda_containers_kernels.cuh"

// System include(s).
#undef NDEBUG
#include <cassert>

int main() {

   // The managed memory resource.
   vecmem::cuda::managed_memory_resource managed_resource;

   // Create an input and an output vector in managed memory.
   vecmem::vector< int > inputvec1( &managed_resource );
   inputvec1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   vecmem::vector< int > outputvec1( inputvec1.size(), &managed_resource );
   assert( inputvec1.size() == outputvec1.size() );

   // Create the array that is used in the linear transformation.
   vecmem::array< int, 2 > constants( managed_resource );
   constants[ 0 ] = 2;
   constants[ 1 ] = 3;

   // Perform a linear transformation using the vecmem vector helper types.
   linearTransform( vecmem::get_data( constants ),
                    vecmem::get_data( inputvec1 ),
                    vecmem::get_data( outputvec1 ) );

   // Check the output.
   for( std::size_t i = 0; i < outputvec1.size(); ++i ) {
      assert( outputvec1.at( i ) ==
              inputvec1.at( i ) * constants[ 0 ] + constants[ 1 ] );
   }

   // The host/device memory resources.
   vecmem::cuda::device_memory_resource device_resource;
   vecmem::cuda::host_memory_resource host_resource;

   // Create input/output vectors on the host.
   vecmem::vector< int > inputvec2( &host_resource );
   inputvec2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   vecmem::vector< int > outputvec2( inputvec2.size(), &host_resource );
   assert( inputvec2.size() == outputvec2.size() );

   // Allocate a device memory block for the output container.
   auto outputvec2host = vecmem::get_data( outputvec2 );
   vecmem::details::vector_buffer< int >
      outputvec2device( outputvec2.size(), device_resource );

   // Perform a linear transformation with explicit memory copies.
   linearTransform( vecmem::get_data( constants ),
                    vecmem::cuda::copy_to_device(
                       vecmem::get_data( inputvec2 ), device_resource ),
                    outputvec2device );
   vecmem::cuda::copy( outputvec2device, outputvec2host );

   // Check the output.
   for( std::size_t i = 0; i < outputvec2.size(); ++i ) {
      assert( outputvec2.at( i ) ==
              inputvec2.at( i ) * constants[ 0 ] + constants[ 1 ] );
   }

   // Return gracefully.
   return 0;
}
