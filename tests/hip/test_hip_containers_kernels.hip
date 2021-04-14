/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// HIP include(s). Note that this needs to come first, as it affects how
// other headers later on include/see system headers like <cassert>.
#include <hip/hip_runtime.h>

// Local include(s).
#include "test_hip_containers_kernels.hpp"
#include "vecmem/containers/const_device_array.hpp"
#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/memory/atomic.hpp"
#include "../../hip/src/utils/hip_error_handling.hpp"

/// Kernel performing a linear transformation using the vector helper types
__global__
void linearTransformKernel( vecmem::data::vector_view< const int > constants,
                            vecmem::data::vector_view< const int > input,
                            vecmem::data::vector_view< int > output ) {

   // Find the current index.
   const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
   if( i >= input.m_size ) {
      return;
   }

   // Create the helper containers.
   const vecmem::const_device_array< int, 2 > constantarray( constants );
   const vecmem::const_device_vector< int > inputvec( input );
   vecmem::device_vector< int > outputvec( output );

   // Perform the linear transformation.
   outputvec.at( i ) = inputvec.at( i ) * constantarray.at( 0 ) +
                       constantarray.at( 1 );
   return;
}

void linearTransform( vecmem::data::vector_view< const int > constants,
                      vecmem::data::vector_view< const int > input,
                      vecmem::data::vector_view< int > output ) {

   // Launch the kernel.
   hipLaunchKernelGGL( linearTransformKernel, 1, input.m_size, 0, nullptr,
                       constants, input, output );
   // Check whether it succeeded to run.
   VECMEM_HIP_ERROR_CHECK( hipGetLastError() );
   VECMEM_HIP_ERROR_CHECK( hipDeviceSynchronize() );
}

/// Kernel performing some basic atomic operations.
__global__
void atomicTransformKernel( std::size_t iterations,
                            vecmem::data::vector_view< int > data ) {

   // Find the current global index.
   const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
   if( i >= ( data.m_size * iterations ) ) {
      return;
   }

   // Get a pointer to the integer that this thread will work on.
   const std::size_t array_index = i % data.m_size;
   assert( array_index < data.m_size );
   int* ptr = data.m_ptr + array_index;

   // Do some simple stuff with it.
   vecmem::atomic< int > a( ptr );
   a.fetch_add( 1 );
   a.fetch_add( 3 );
   return;
}

void atomicTransform( std::size_t iterations,
                      vecmem::data::vector_view< int > vec ) {

   // Launch the kernel.
   hipLaunchKernelGGL( atomicTransformKernel, iterations, vec.m_size, 0,
                       nullptr, iterations, vec );
   // Check whether it succeeded to run.
   VECMEM_HIP_ERROR_CHECK( hipGetLastError() );
   VECMEM_HIP_ERROR_CHECK( hipDeviceSynchronize() );
}
