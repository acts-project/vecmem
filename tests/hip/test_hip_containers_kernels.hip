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
#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "../../hip/src/utils/hip_error_handling.hpp"

/// Kernel performing a linear transformation using the vector helper types
__global__
void linearTransformKernel( std::size_t size, const int* input, int* output ) {

   // Find the current index.
   const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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

void linearTransform( std::size_t size, const int* input, int* output ) {

   // Launch the kernel.
   hipLaunchKernelGGL( linearTransformKernel, 1, size, 0, nullptr,
                       size, input, output );
   VECMEM_HIP_ERROR_CHECK( hipGetLastError() );
   VECMEM_HIP_ERROR_CHECK( hipDeviceSynchronize() );
}
