/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/array.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "test_cuda_containers_kernels.cuh"

// System include(s).
#undef NDEBUG
#include <cassert>

int main() {
   vecmem::cuda::managed_memory_resource resource;

   // Create an input and an output vector.
   vecmem::vector<int> inputvec(&resource);
   inputvec = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

   vecmem::vector<int> outputvec(&resource);
   outputvec.resize(10);

   assert( inputvec.size() == outputvec.size() );

   // Create the array that is used in the linear transformation.
   vecmem::array< int, 2 > constants( resource );
   constants[ 0 ] = 2;
   constants[ 1 ] = 3;

   // Perform a linear transformation using the vecmem vector helper types.
   linearTransform( vecmem::get_data( constants ), vecmem::get_data( inputvec ),
                    vecmem::get_data( outputvec ) );

   // Check the output.
   for( std::size_t i = 0; i < outputvec.size(); ++i ) {
      assert( outputvec.at( i ) ==
              inputvec.at( i ) * constants[ 0 ] + constants[ 1 ] );
   }

   // Return gracefully.
   return 0;
}
