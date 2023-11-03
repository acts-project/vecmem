/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// HIP include(s). Note that this needs to come first, as it affects how
// other headers later on include/see system headers like <cassert>.
#include <hip/hip_runtime.h>

// Local include(s).
#include "../../hip/src/utils/hip_error_handling.hpp"
#include "test_hip_edm_kernels.hpp"

/// Kernel modifying the data in a container
__global__ void edmModifyKernel(vecmem::testing::simple_container::view view) {

    // Get the thread index.
    const unsigned int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    // Run the modification using the helper function.
    vecmem::testing::simple_container::device device{view};
    modify(i, device);
}

void edmModify(vecmem::testing::simple_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.size() + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(edmModifyKernel, gridSize, blockSize, 0, nullptr, view);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel filling data into a container
__global__ void edmFillKernel(vecmem::testing::simple_container::view view) {

    // Get the thread index.
    const unsigned int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

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
    hipLaunchKernelGGL(edmFillKernel, gridSize, blockSize, 0, nullptr, view);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}
