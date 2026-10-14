/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/vector.hpp"

// System include(s).
#include <numeric>

/// Test de-allocating memory while it is still in use on the stream
TEST_P(memory_resource_test_stream_ordered, deallocate_while_in_use) {

    // Large enough for the copies to still be running at the de-allocation.
    static constexpr unsigned int SIZE = 16 * 1024 * 1024;

    vecmem::copy& copy = *(std::get<0>(GetParam()));
    vecmem::memory_resource& resource = *(std::get<2>(GetParam()));
    vecmem::memory_resource& host_resource = *(std::get<3>(GetParam()));

    vecmem::vector<int> input(SIZE, &host_resource);
    std::iota(input.begin(), input.end(), 0);
    vecmem::vector<int> output(SIZE, 0, &host_resource);

    // Copy through device memory, de-allocating it right away.
    {
        vecmem::data::vector_buffer<int> buffer(SIZE, resource);
        copy(vecmem::get_data(input), buffer,
             vecmem::copy::type::host_to_device)
            ->ignore();
        copy(buffer, vecmem::get_data(output),
             vecmem::copy::type::device_to_host)
            ->ignore();
    }

    // Overwrite a new allocation, and wait for everything to finish.
    vecmem::data::vector_buffer<int> buffer(SIZE, resource);
    copy.memset(buffer, 0xff)->wait();

    EXPECT_EQ(input, output);
}
