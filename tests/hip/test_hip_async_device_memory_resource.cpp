/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/hip/async_device_memory_resource.hpp"
#include "vecmem/memory/hip/host_memory_resource.hpp"
#include "vecmem/utils/hip/async_copy.hpp"
#include "vecmem/utils/hip/stream_wrapper.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <numeric>
#include <vector>

/// Test that (de-)allocations are queued on the stream, without synchronising
TEST(hip_async_device_memory_resource_test, deallocate_while_in_use) {

    static constexpr unsigned int SIZE = 1024;

    vecmem::hip::stream_wrapper test_stream;
    vecmem::hip::async_device_memory_resource resource(test_stream);
    vecmem::hip::async_copy copy(test_stream);
    vecmem::hip::host_memory_resource host_resource;
    hipStream_t hip_stream = static_cast<hipStream_t>(test_stream.stream());

    vecmem::vector<int> input(SIZE, &host_resource);
    std::iota(input.begin(), input.end(), 0);
    vecmem::vector<int> output(SIZE, 0, &host_resource);

    // Allocate, use and free device memory while capturing the stream.
    ASSERT_EQ(hipStreamBeginCapture(hip_stream, hipStreamCaptureModeGlobal),
              hipSuccess);
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
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipStreamEndCapture(hip_stream, &graph), hipSuccess);

    // Check that all operations were recorded, in any order.
    std::size_t n_nodes = 0;
    ASSERT_EQ(hipGraphGetNodes(graph, nullptr, &n_nodes), hipSuccess);
    std::vector<hipGraphNode_t> nodes(n_nodes);
    ASSERT_EQ(hipGraphGetNodes(graph, nodes.data(), &n_nodes), hipSuccess);
    std::vector<hipGraphNodeType> types(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        ASSERT_EQ(hipGraphNodeGetType(nodes[i], &types[i]), hipSuccess);
    }
    std::sort(types.begin(), types.end());
    std::vector<hipGraphNodeType> expected = {
        hipGraphNodeTypeMemAlloc, hipGraphNodeTypeMemcpy,
        hipGraphNodeTypeMemcpy, hipGraphNodeTypeMemFree};
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(types, expected);

    ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
}
