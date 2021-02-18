/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "../../cuda/src/utils/cuda_error_handling.hpp"

#include <cstddef>
#include <stdexcept>

/// Kernel performing a linear transformation using the vector helper types
__global__
void linearTransformKernel( std::size_t size, const int* input, int* output ) {

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

void linearTransform( std::size_t size, const int* input, int* output ) {
   linearTransformKernel<<< 1, size >>>( size, input, output );

   VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
   VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
