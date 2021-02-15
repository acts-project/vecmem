/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/allocators/allocator.hpp"
#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/memory/memory_manager.hpp"
#include "vecmem/memory/cuda/direct_memory_manager.hpp"
#include "vecmem/memory/cuda/resources/resources.hpp"
#include "vecmem/utils/cuda_error_handling.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <vector>

/// Custom vector type used on the host in the tests
template<typename T>
using managed_vector = std::vector<T, std::pmr::polymorphic_allocator<T>>;

/// Helper function for creating an "input vector".
managed_vector< int > make_input_vector() {
   managed_vector<int> vec(vecmem::memory::resources::get_terminal_cuda_managed_resource());

   vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

   return vec;
}

/// Helper function for creating an "output vector".
managed_vector< int > make_output_vector() {
   managed_vector<int> vec(vecmem::memory::resources::get_terminal_cuda_managed_resource());

   vec.resize(10);

   return vec;
}

/// Kernel performing a linear transformation using the vector helper types
__global__
void testLinearTransform( std::size_t size, const int* input, int* output ) {

   // Find the current index.
   const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if( i >= size ) {
      return;
   }

   // Create the helper vectors.
   const vecmem::const_device_vector< int > inputvec( size, input );
   vecmem::device_vector< int > outputvec( size, output );

   // Perform the linear transformation.
   outputvec.at( i ) = 3 + inputvec.at( i ) * 2;
   return;
}

int main() {
   // Create an input and an output vector.
   auto inputvec = make_input_vector();
   auto outputvec = make_output_vector();
   assert( inputvec.size() == outputvec.size() );

   // Perform a linear transformation using the vecmem vector helper types.
   testLinearTransform<<< 1, inputvec.size() >>>( inputvec.size(),
                                                  inputvec.data(),
                                                  outputvec.data() );
   VECMEM_CUDA_ERROR_CHECK( cudaGetLastError() );
   VECMEM_CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

   // Check the output.
   for( std::size_t i = 0; i < outputvec.size(); ++i ) {
      assert( outputvec.at( i ) == inputvec.at( i ) * 2 + 3 );
   }

   // Return gracefully.
   return 0;
}
