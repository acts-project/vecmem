/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "test_cuda_containers_kernels.cuh"
#include "vecmem/containers/const_device_array.hpp"
#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "../../cuda/src/utils/cuda_error_handling.hpp"

/// Kernel performing a linear transformation using the vector helper types
__global__
void linearTransformKernel( vecmem::details::vector_view< const int > constants,
                            vecmem::details::vector_view< const int > input,
                            vecmem::details::vector_view< int > output ) {

   // Find the current index.
   const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
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
   linearTransformKernel<<< 1, input.m_size >>>( constants, input, output );
   // Check whether it succeeded to run.
   VECMEM_CUDA_ERROR_CHECK( cudaGetLastError() );
   VECMEM_CUDA_ERROR_CHECK( cudaDeviceSynchronize() );
}
