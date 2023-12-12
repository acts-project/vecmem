/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "jagged_soa_container_helpers.hpp"
#include "simple_soa_container_helpers.hpp"

template <typename CONTAINER>
void soa_copy_tests_base<CONTAINER>::host_to_fixed_device_to_host_direct(
    const soa_copy_test_parameters& params) {

    // Extract the parameters.
    vecmem::memory_resource& host_mr = std::get<0>(params);
    vecmem::memory_resource& device_mr = std::get<1>(params);
    vecmem::copy& device_copy = std::get<3>(params);

    // Create the "input" host container.
    typename CONTAINER::host input{host_mr};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create the (fixed sized) device buffer.
    typename CONTAINER::buffer device_buffer;
    vecmem::testing::make_buffer(device_buffer, device_mr, host_mr,
                                 vecmem::data::buffer_type::fixed_size);
    device_copy.setup(device_buffer);

    // Copy the data to the device.
    device_copy(vecmem::get_data(input), device_buffer,
                vecmem::copy::type::host_to_device);

    // Create the target host container.
    typename CONTAINER::host target{host_mr};

    // Copy the data back to the host.
    device_copy(device_buffer, target, vecmem::copy::type::device_to_host)
        ->wait();

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));
}

template <typename CONTAINER>
void soa_copy_tests_base<CONTAINER>::host_to_fixed_device_to_host_optimal(
    const soa_copy_test_parameters& params) {

    // Extract the parameters.
    vecmem::memory_resource& host_mr = std::get<0>(params);
    vecmem::memory_resource& device_mr = std::get<1>(params);
    vecmem::copy& host_copy = std::get<2>(params);
    vecmem::copy& device_copy = std::get<3>(params);

    // Create the "input" host container.
    typename CONTAINER::host input{host_mr};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create a (fixed sized) host buffer, to stage the data into.
    typename CONTAINER::buffer host_buffer1;
    vecmem::testing::make_buffer(host_buffer1, host_mr, host_mr,
                                 vecmem::data::buffer_type::fixed_size);
    host_copy.setup(host_buffer1);

    // Stage the data into the host buffer.
    host_copy(vecmem::get_data(input), host_buffer1);

    // Create the (fixed sized) device buffer.
    typename CONTAINER::buffer device_buffer;
    vecmem::testing::make_buffer(device_buffer, device_mr, host_mr,
                                 vecmem::data::buffer_type::fixed_size);
    device_copy.setup(device_buffer);

    // Copy the data from the host buffer to the device buffer.
    device_copy(host_buffer1, device_buffer,
                vecmem::copy::type::host_to_device);

    // Create a (fixed sized) host buffer, to stage the data back into.
    typename CONTAINER::buffer host_buffer2;
    vecmem::testing::make_buffer(host_buffer2, host_mr, host_mr,
                                 vecmem::data::buffer_type::fixed_size);
    host_copy.setup(host_buffer2);

    // Copy the data from the device buffer to the host buffer.
    device_copy(device_buffer, host_buffer2, vecmem::copy::type::device_to_host)
        ->wait();

    // Create the target host container.
    typename CONTAINER::host target{host_mr};

    // Copy the data from the host buffer to the target.
    host_copy(host_buffer2, target);

    // Compare the relevant objects.
    vecmem::testing::compare(vecmem::get_data(input),
                             vecmem::get_data(host_buffer2));
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));
}

template <typename CONTAINER>
void soa_copy_tests_base<CONTAINER>::host_to_resizable_device_to_host(
    const soa_copy_test_parameters& params) {

    // Extract the parameters.
    vecmem::memory_resource& host_mr = std::get<0>(params);
    vecmem::memory_resource& device_mr = std::get<1>(params);
    vecmem::copy& device_copy = std::get<3>(params);

    // Create the "input" host container.
    typename CONTAINER::host input{host_mr};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create the (resizable) device buffer.
    typename CONTAINER::buffer device_buffer;
    vecmem::testing::make_buffer(device_buffer, device_mr, host_mr,
                                 vecmem::data::buffer_type::resizable);
    device_copy.setup(device_buffer);

    // Copy the data to the device.
    device_copy(vecmem::get_data(input), device_buffer,
                vecmem::copy::type::host_to_device);

    // Create the target host container.
    typename CONTAINER::host target{host_mr};

    // Copy the data back to the host.
    device_copy(device_buffer, target, vecmem::copy::type::device_to_host)
        ->wait();

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));
}

template <typename CONTAINER>
void soa_copy_tests_base<CONTAINER>::
    host_to_fixed_device_to_resizable_device_to_host(
        const soa_copy_test_parameters& params) {

    // Extract the parameters.
    vecmem::memory_resource& host_mr = std::get<0>(params);
    vecmem::memory_resource& device_mr = std::get<1>(params);
    vecmem::copy& device_copy = std::get<3>(params);

    // Create the "input" host container.
    typename CONTAINER::host input{host_mr};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create the (fixed sized) device buffer.
    typename CONTAINER::buffer device_buffer1;
    vecmem::testing::make_buffer(device_buffer1, device_mr, host_mr,
                                 vecmem::data::buffer_type::fixed_size);
    device_copy.setup(device_buffer1);

    // Copy the data to the device.
    device_copy(vecmem::get_data(input), device_buffer1,
                vecmem::copy::type::host_to_device);

    // Create the (resizable) device buffer.
    typename CONTAINER::buffer device_buffer2;
    vecmem::testing::make_buffer(device_buffer2, device_mr, host_mr,
                                 vecmem::data::buffer_type::resizable);

    // Copy the data from the fixed sized device buffer to the resizable one.
    device_copy(device_buffer1, device_buffer2,
                vecmem::copy::type::device_to_device);

    // Create the target host container.
    typename CONTAINER::host target{host_mr};

    // Copy the data back to the host.
    device_copy(device_buffer2, target, vecmem::copy::type::device_to_host)
        ->wait();

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));
}

TEST_P(soa_copy_tests_simple, host_to_fixed_device_to_host_direct) {

    host_to_fixed_device_to_host_direct(GetParam());
}

TEST_P(soa_copy_tests_simple, host_to_fixed_device_to_host_optimal) {

    host_to_fixed_device_to_host_optimal(GetParam());
}

TEST_P(soa_copy_tests_simple, host_to_resizable_device_to_host) {

    host_to_resizable_device_to_host(GetParam());
}

TEST_P(soa_copy_tests_simple,
       host_to_fixed_device_to_resizable_device_to_host) {

    host_to_fixed_device_to_resizable_device_to_host(GetParam());
}

TEST_P(soa_copy_tests_jagged, host_to_fixed_device_to_host_direct) {

    host_to_fixed_device_to_host_direct(GetParam());
}

TEST_P(soa_copy_tests_jagged, host_to_fixed_device_to_host_optimal) {

    host_to_fixed_device_to_host_optimal(GetParam());
}

TEST_P(soa_copy_tests_jagged, host_to_resizable_device_to_host) {

    host_to_resizable_device_to_host(GetParam());
}

TEST_P(soa_copy_tests_jagged,
       host_to_fixed_device_to_resizable_device_to_host) {

    host_to_fixed_device_to_resizable_device_to_host(GetParam());
}
