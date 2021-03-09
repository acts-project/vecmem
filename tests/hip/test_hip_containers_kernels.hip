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
#include "../../hip/src/utils/hip_error_handling.hpp"

/// Kernel performing a linear transformation using the vector helper types
__global__
void linearTransformKernel( vecmem::details::vector_view< const int > constants,
                            vecmem::details::vector_view< const int > input,
                            vecmem::details::vector_view< int > output ) {

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

void linearTransform( vecmem::details::vector_view< const int > constants,
                      vecmem::details::vector_view< const int > input,
                      vecmem::details::vector_view< int > output ) {

   // Launch the kernel.
   hipLaunchKernelGGL( linearTransformKernel, 1, input.m_size, 0, nullptr,
                       constants, input, output );
   // Check whether it succeeded to run.
   VECMEM_HIP_ERROR_CHECK( hipGetLastError() );
   VECMEM_HIP_ERROR_CHECK( hipDeviceSynchronize() );
}
