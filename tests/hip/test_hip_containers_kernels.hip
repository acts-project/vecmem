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
#include "../../hip/src/utils/hip_error_handling.hpp"
#include "test_hip_containers_kernels.hpp"
#include "vecmem/containers/const_device_array.hpp"
#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/static_array.hpp"
#include "vecmem/memory/atomic.hpp"

/// Kernel performing a linear transformation using the vector helper types
__global__ void linearTransformKernel(
    vecmem::data::vector_view<const int> constants,
    vecmem::data::vector_view<const int> input,
    vecmem::data::vector_view<int> output) {

    // Find the current index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= input.size()) {
        return;
    }

    // Create the helper containers.
    const vecmem::const_device_array<int, 2> constantarray1(constants);
    const vecmem::static_array<int, 2> constantarray2 = {constantarray1[0],
                                                         constantarray1[1]};
    const vecmem::const_device_vector<int> inputvec(input);
    vecmem::device_vector<int> outputvec(output);

    // Perform the linear transformation.
    outputvec.at(i) = inputvec.at(i) * constantarray1.at(0) + constantarray2[1];
    return;
}

void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output) {

    // Launch the kernel.
    hipLaunchKernelGGL(linearTransformKernel, 1, input.size(), 0, nullptr,
                       constants, input, output);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel performing a linear transformation using the vector helper types
__global__ void linearTransformKernel(
    vecmem::data::vector_view<const int> constants,
    vecmem::data::jagged_vector_view<const int> input,
    vecmem::data::jagged_vector_view<int> output) {

    // Find the current index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= input.m_size) {
        return;
    }

    // Create the helper containers.
    const vecmem::device_array<const int, 2> constantarray(constants);
    const vecmem::jagged_device_vector<const int> inputvec(input);
    vecmem::jagged_device_vector<int> outputvec(output);

    // A little sanity check.
    assert(inputvec.at(i).size() == outputvec.at(i).size());

    // Perform the requested linear transformation on all elements of a given
    // "internal vector".
    for (std::size_t j = 0; j < inputvec[i].size(); ++j) {
        outputvec[i][j] = inputvec[i][j] * constantarray[0] + constantarray[1];
    }
    __syncthreads();

    // Now exercise the jagged vector iterators in a bit of an elaborate
    // operation.
    for (auto itr1 = outputvec.rbegin(); itr1 != outputvec.rend(); ++itr1) {
        if ((outputvec[i].size() > 0) && (itr1->size() > 1)) {
            // Iterate over all inner vectors, skipping the first elements.
            // Since those are being updated at the same time, by other threads.
            for (auto itr2 = itr1->rbegin(); itr2 != (itr1->rend() - 1);
                 ++itr2) {
                outputvec[i].front() += *itr2;
            }
        }
    }
}

void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::jagged_vector_view<const int> input,
                     vecmem::data::jagged_vector_view<int> output) {

    // A sanity check.
    assert(input.m_size == output.m_size);

    // Launch the kernel.
    hipLaunchKernelGGL(linearTransformKernel, 1, input.m_size, 0, nullptr,
                       constants, input, output);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel performing some basic atomic operations.
__global__ void atomicTransformKernel(std::size_t iterations,
                                      vecmem::data::vector_view<int> data) {

    // Find the current global index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= (data.size() * iterations)) {
        return;
    }

    // Get a pointer to the integer that this thread will work on.
    const std::size_t array_index = i % data.size();
    assert(array_index < data.size());
    int* ptr = data.ptr() + array_index;

    // Do some simple stuff with it.
    vecmem::atomic<int> a(ptr);
    a.fetch_add(1);
    a.fetch_add(3);
    return;
}

void atomicTransform(std::size_t iterations,
                     vecmem::data::vector_view<int> vec) {

    // Launch the kernel.
    hipLaunchKernelGGL(atomicTransformKernel, iterations, vec.size(), 0,
                       nullptr, iterations, vec);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel filtering the input vector elements into the output vector
__global__ void filterTransformKernel(
    vecmem::data::vector_view<const int> input,
    vecmem::data::vector_view<int> output) {

    // Find the current global index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= input.size()) {
        return;
    }

    // Set up the vector objects.
    const vecmem::const_device_vector<int> inputvec(input);
    vecmem::device_vector<int> outputvec(output);

    // Add this thread's element, if it passes the selection.
    const int element = inputvec.at(i);
    if (element > 10) {
        outputvec.push_back(element);
    }
    return;
}

void filterTransform(vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output) {

    // Launch the kernel.
    hipLaunchKernelGGL(filterTransformKernel, 1, input.size(), 0, nullptr,
                       input, output);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel filtering the input vector elements into the output vector
__global__ void filterTransformKernel(
    vecmem::data::jagged_vector_view<const int> input,
    vecmem::data::jagged_vector_view<int> output) {

    // Find the current indices.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= input.m_size) {
        return;
    }
    const std::size_t j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (j >= input.m_ptr[i].size()) {
        return;
    }

    // Set up the vector objects.
    const vecmem::jagged_device_vector<const int> inputvec(input);
    vecmem::jagged_device_vector<int> outputvec(output);

    // Keep just the odd elements.
    const int value = inputvec[i][j];
    if ((value % 2) != 0) {
        outputvec.at(i).push_back(value);
    }
    return;
}

void filterTransform(vecmem::data::jagged_vector_view<const int> input,
                     std::size_t max_vec_size,
                     vecmem::data::jagged_vector_view<int> output) {

    // Launch the kernel.
    hipLaunchKernelGGL(filterTransformKernel, 1,
                       dim3(input.m_size, max_vec_size), 0, nullptr, input,
                       output);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}
