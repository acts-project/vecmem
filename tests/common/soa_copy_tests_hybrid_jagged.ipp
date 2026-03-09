/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "jagged_soa_container.hpp"
#include "jagged_soa_container_helpers.hpp"
#include "soa_copy_tests_hybrid_jagged.hpp"

// System include(s).
#include <algorithm>

vecmem::copy& soa_copy_tests_hybrid_jagged::main_copy() {
    return *(std::get<0>(GetParam()));
}

vecmem::copy& soa_copy_tests_hybrid_jagged::host_copy() {
    return *(std::get<1>(GetParam()));
}

vecmem::memory_resource& soa_copy_tests_hybrid_jagged::main_mr() {
    return *(std::get<2>(GetParam()));
}

vecmem::memory_resource& soa_copy_tests_hybrid_jagged::host_mr() {
    return *(std::get<3>(GetParam()));
}

/// Test for the host->resizable device->host copy
TEST_P(soa_copy_tests_hybrid_jagged, host_to_resizable_device_to_host) {

    // Create the "input" host container.
    vecmem::testing::jagged_soa_container::host input{host_mr()};

    // Fill it with some (uneven) data.
    constexpr bool UNEVEN = true;
    vecmem::testing::fill(input, UNEVEN);

    // Create a resizable device buffer that would be able to hold this
    // data.
    vecmem::testing::jagged_soa_container::buffer device_buffer{
        vecmem::edm::get_capacities(vecmem::get_data(input)), main_mr(),
        &(host_mr()), vecmem::data::buffer_type::resizable};
    main_copy().setup(device_buffer)->wait();

    // Copy the data to the device.
    const vecmem::testing::jagged_soa_container::data input_data =
        vecmem::get_data(input);
    main_copy()(vecmem::get_data(input_data), device_buffer,
                vecmem::copy::type::host_to_device)
        ->wait();

    // Create the target host container.
    vecmem::testing::jagged_soa_container::host target{host_mr()};

    // Copy the data back to the host.
    main_copy()(device_buffer, target, vecmem::copy::type::device_to_host)
        ->wait();

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));

    // Copy the data back to the host, using the "to" function.
    auto target2 = main_copy().to(device_buffer, host_mr(), nullptr,
                                  vecmem::copy::type::device_to_host);
    EXPECT_NE(target2.size().ptr(), nullptr);
    EXPECT_EQ(device_buffer.capacity(), target2.capacity());
    EXPECT_EQ(main_copy().get_size(device_buffer),
              host_copy().get_size(target2));

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(input),
                             vecmem::get_data(target2));

    // Make a host version of this latter buffer.
    vecmem::testing::jagged_soa_container::host target3{host_mr()};
    main_copy()(target2, target3, vecmem::copy::type::device_to_host)->wait();

    // Compare this to thr original as well.
    vecmem::testing::compare(vecmem::get_data(input),
                             vecmem::get_data(target3));
}
