/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../cuda/src/utils/cuda_error_handling.hpp"
#include "test_cuda_edm_kernels.hpp"

/// Kernel modifying the data in a container
__global__ void edmModifyKernel(vecmem::testing::simple_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::simple_container::device device{view};
    modify(i, device);
}

void edmModify(vecmem::testing::simple_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    edmModifyKernel<<<gridSize, blockSize>>>(view);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel filling data into a container
__global__ void edmFillKernel(vecmem::testing::simple_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::simple_container::device device{view};
    if (i < device.capacity()) {
        vecmem::testing::simple_container::device::size_type ii =
            device.push_back_default();
        fill(ii, device);
    }
}

void edmFill(vecmem::testing::simple_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    edmFillKernel<<<gridSize, blockSize>>>(view);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
