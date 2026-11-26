/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/async_device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/utils/cuda/async_copy.hpp"
#include "vecmem/utils/cuda/stream_wrapper.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <numeric>
#include <vector>

/// Test that (de-)allocations are queued on the stream, without synchronising
TEST(cuda_async_device_memory_resource_test, deallocate_while_in_use) {

    static constexpr unsigned int SIZE = 1024;

    vecmem::cuda::stream_wrapper test_stream;
    vecmem::cuda::async_device_memory_resource resource(test_stream);
    vecmem::cuda::async_copy copy(test_stream);
    vecmem::cuda::host_memory_resource host_resource;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(test_stream.stream());

    vecmem::vector<int> input(SIZE, &host_resource);
    std::iota(input.begin(), input.end(), 0);
    vecmem::vector<int> output(SIZE, 0, &host_resource);

    // Allocate, use and free device memory while capturing the stream.
    ASSERT_EQ(cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeGlobal),
              cudaSuccess);
    EXPECT_NO_THROW({
        void* ptr = resource.allocate(SIZE * sizeof(int));
        vecmem::data::vector_view<int> buffer(SIZE, static_cast<int*>(ptr));
        copy(vecmem::get_data(input), buffer,
             vecmem::copy::type::host_to_device)
            ->ignore();
        copy(buffer, vecmem::get_data(output),
             vecmem::copy::type::device_to_host)
            ->ignore();
        resource.deallocate(ptr, SIZE * sizeof(int));
    });
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(cuda_stream, &graph), cudaSuccess);

    // Check that all operations were recorded, in any order.
    std::size_t n_nodes = 0;
    ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &n_nodes), cudaSuccess);
    std::vector<cudaGraphNode_t> nodes(n_nodes);
    ASSERT_EQ(cudaGraphGetNodes(graph, nodes.data(), &n_nodes), cudaSuccess);
    std::vector<cudaGraphNodeType> types(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        ASSERT_EQ(cudaGraphNodeGetType(nodes[i], &types[i]), cudaSuccess);
    }
    std::sort(types.begin(), types.end());
    std::vector<cudaGraphNodeType> expected = {
        cudaGraphNodeTypeMemAlloc, cudaGraphNodeTypeMemcpy,
        cudaGraphNodeTypeMemcpy, cudaGraphNodeTypeMemFree};
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(types, expected);

    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}
