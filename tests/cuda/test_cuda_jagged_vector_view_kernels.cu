/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "test_cuda_jagged_vector_view_kernels.cuh"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "../../cuda/src/utils/cuda_error_handling.hpp"

__global__
void doubleJaggedKernel(
    vecmem::data::jagged_vector_view<int> _jag
) {
    const std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    vecmem::jagged_device_vector<int> jag(_jag);

    if (t >= jag.size()) {
        return;
    }

    for (std::size_t i = 0; i < jag.at(t).size(); ++i) {
        jag.at(t).at(i) *= 2;
    }
    __syncthreads();
    // Iterate over all outer vectors.
    for( auto itr1 = jag.rbegin(); itr1 != jag.rend(); ++itr1 ) {
        if( ( jag[ t ].size() > 0 ) && ( itr1->size() > 1 ) ) {
            // Iterate over all inner vectors, skipping the first elements.
            // Since those are being updated at the same time, by other threads.
            for( auto itr2 = itr1->rbegin(); itr2 != ( itr1->rend() - 1 );
                 ++itr2 ) {
                jag[ t ].front() += *itr2;
            }
        }
    }
}

void doubleJagged(
    vecmem::data::jagged_vector_view<int> & jag
) {
    doubleJaggedKernel<<<1, jag.m_size>>>(jag);

    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
