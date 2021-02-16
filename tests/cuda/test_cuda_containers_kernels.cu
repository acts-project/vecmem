#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/utils/cuda_error_handling.hpp"

/// Kernel performing a linear transformation using the vector helper types
__global__
void linearTransform( std::size_t size, const int* input, int* output ) {

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

void doLinearTransform(std::size_t size, const int* input, int* output) {
   // Perform a linear transformation using the vecmem vector helper types.
   linearTransform<<<1, size>>>(size, input, output);

   VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
   VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}