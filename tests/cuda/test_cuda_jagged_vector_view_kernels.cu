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
}

void doubleJagged(
    vecmem::data::jagged_vector_view<int> & jag
) {
    doubleJaggedKernel<<<1, jag.m_size>>>(jag);

    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
