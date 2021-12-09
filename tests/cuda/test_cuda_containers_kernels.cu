/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../cuda/src/utils/cuda_error_handling.hpp"
#include "../../cuda/src/utils/cuda_wrappers.hpp"
#include "test_cuda_containers_kernels.cuh"
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
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
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
    outputvec.at(i) =
        inputvec.at(i) * constantarray1.at(0) + vecmem::get<1>(constantarray2);
    return;
}

void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output) {

    // Launch the kernel.
    linearTransformKernel<<<1, input.size()>>>(constants, input, output);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output,
                     const vecmem::cuda::stream_wrapper& stream) {

    // Launch the kernel.
    linearTransformKernel<<<1, input.size(), 0,
                            vecmem::cuda::details::get_stream(stream)>>>(
        constants, input, output);
    // Check whether it succeeded to launch.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
}

/// Kernel performing some basic atomic operations.
__global__ void atomicTransformKernel(std::size_t iterations,
                                      vecmem::data::vector_view<int> data) {

    // Find the current global index.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (data.size() * iterations)) {
        return;
    }

    // Get a pointer to the integer that this thread will work on.
    const std::size_t array_index = i % data.size();
    assert(array_index < data.size());
    int* ptr = data.ptr() + array_index;

    // Do some simple stuff with it.
    vecmem::atomic<int> a(ptr);
    a.fetch_add(4);
    a.fetch_sub(2);
    a.fetch_and(0xffffffff);
    a.fetch_or(0x00000000);
    return;
}

void atomicTransform(unsigned int iterations,
                     vecmem::data::vector_view<int> vec) {

    // Launch the kernel.
    atomicTransformKernel<<<iterations, vec.size()>>>(iterations, vec);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel filtering the input vector elements into the output vector
__global__ void filterTransformKernel(
    vecmem::data::vector_view<const int> input,
    vecmem::data::vector_view<int> output) {

    // Find the current global index.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
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
    filterTransformKernel<<<1, input.size()>>>(input, output);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel filtering the input vector elements into the output vector
__global__ void filterTransformKernel(
    vecmem::data::jagged_vector_view<const int> input,
    vecmem::data::jagged_vector_view<int> output) {

    // Find the current indices.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input.m_size) {
        return;
    }
    const std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;
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
                     unsigned int max_vec_size,
                     vecmem::data::jagged_vector_view<int> output) {

    // Launch the kernel.
    dim3 dimensions(static_cast<unsigned int>(input.m_size), max_vec_size);
    filterTransformKernel<<<1, dimensions>>>(input, output);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel filling a jagged vector to its capacity
__global__ void fillTransformKernel(
    vecmem::data::jagged_vector_view<int> vec_data) {

    // Find the current index.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vec_data.m_size) {
        return;
    }

    // Create a device vector on top of the view.
    vecmem::jagged_device_vector<int> vec(vec_data);

    // Fill the vectors to their capacity.
    while (vec[i].size() < vec[i].capacity()) {
        vec[i].push_back(1);
    }
}

void fillTransform(vecmem::data::jagged_vector_view<int> vec) {

    // Launch the kernel
    fillTransformKernel<<<static_cast<unsigned int>(vec.m_size), 1>>>(vec);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void readArrayKernel(
    vecmem::static_array<vecmem::data::vector_view<int>, 3> arr_vec) {

    vecmem::device_vector<int> vec(arr_vec[0]);

    // It's OK
    printf("%d", vec.size());

    // this doesn't work
    printf("%d", vec[0]);
}

void readArray(
    vecmem::static_array<vecmem::data::vector_view<int>, 3> arr_vec) {

    readArrayKernel<<<1, 1>>>(arr_vec);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}