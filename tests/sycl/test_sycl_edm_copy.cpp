/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/edm_simple_container.hpp"
#include "../common/edm_simple_container_helpers.hpp"
#include "vecmem/memory/sycl/device_memory_resource.hpp"
#include "vecmem/memory/sycl/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/sycl/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test fixture for the SYCL EDM copy tests
class sycl_edm_copy_test : public testing::Test {

protected:
    /// SYCL queue to use
    vecmem::sycl::queue_wrapper m_queue;

    /// Host memory resource
    vecmem::sycl::host_memory_resource m_host_mr{m_queue};
    /// Device memory resource
    vecmem::sycl::device_memory_resource m_device_mr{m_queue};

    /// Helper object for performing (SYCL) memory copies
    vecmem::sycl::copy m_sycl_copy{m_queue};
    /// Helper object for performing (host) memory copies
    vecmem::copy m_host_copy;

};  // class sycl_edm_copy_test

/// Convenience type declaration
using test_edm = vecmem::testing::simple_container;

TEST_F(sycl_edm_copy_test, host_to_fixed_device_to_host_simple) {

    // Create the "input" host container.
    test_edm::host input{m_host_mr};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create the (fixed sized) device buffer.
    test_edm::buffer device_buffer{
        static_cast<test_edm::buffer::size_type>(input.size()), m_device_mr,
        vecmem::data::buffer_type::fixed_size};
    m_sycl_copy.setup(device_buffer);

    // Copy the data to the device.
    m_sycl_copy(vecmem::get_data(input), device_buffer,
                vecmem::copy::type::host_to_device);

    // Create the target host container.
    test_edm::host target{m_host_mr};

    // Copy the data back to the host.
    m_sycl_copy(device_buffer, target, vecmem::copy::type::device_to_host);

    // Compare the two.
    vecmem::testing::compare(input, target);
}

TEST_F(sycl_edm_copy_test, host_to_fixed_device_to_host_optimal) {

    // Create the "input" host container.
    test_edm::host input{m_host_mr};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create a (fixed sized) host buffer, to stage the data into.
    test_edm::buffer host_buffer1{
        static_cast<test_edm::buffer::size_type>(input.size()), m_host_mr,
        vecmem::data::buffer_type::fixed_size};
    m_host_copy.setup(host_buffer1);

    // Stage the data into the host buffer.
    m_host_copy(vecmem::get_data(input), host_buffer1);

    // Create the (fixed sized) device buffer.
    test_edm::buffer device_buffer{host_buffer1.capacity(), m_device_mr,
                                   vecmem::data::buffer_type::fixed_size};
    m_sycl_copy.setup(device_buffer);

    // Copy the data from the host buffer to the device buffer.
    m_sycl_copy(host_buffer1, device_buffer,
                vecmem::copy::type::host_to_device);

    // Create a (fixed sized) host buffer, to stage the data back into.
    test_edm::buffer host_buffer2{host_buffer1.capacity(), m_host_mr,
                                  vecmem::data::buffer_type::fixed_size};
    m_host_copy.setup(host_buffer2);

    // Copy the data from the device buffer to the host buffer.
    m_sycl_copy(device_buffer, host_buffer2,
                vecmem::copy::type::device_to_host);

    // Create the target host container.
    test_edm::host target{m_host_mr};

    // Copy the data from the host buffer to the target.
    m_host_copy(host_buffer2, target);

    // Compare the two.
    vecmem::testing::compare(input, target);
}

TEST_F(sycl_edm_copy_test, host_to_resizable_device_to_host_simple) {

    // Create the "input" host container.
    test_edm::host input{m_host_mr};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create the (fixed sized) device buffer.
    test_edm::buffer device_buffer{
        static_cast<test_edm::buffer::size_type>(input.size()), m_device_mr,
        vecmem::data::buffer_type::resizable};
    m_sycl_copy.setup(device_buffer);

    // Copy the data to the device.
    m_sycl_copy(vecmem::get_data(input), device_buffer,
                vecmem::copy::type::host_to_device);

    // Create the target host container.
    test_edm::host target{m_host_mr};

    // Copy the data back to the host.
    m_sycl_copy(device_buffer, target, vecmem::copy::type::device_to_host);

    // Compare the two.
    vecmem::testing::compare(input, target);
}

TEST_F(sycl_edm_copy_test, host_to_resizable_device_to_host_optimal) {

    // Create the "input" host container.
    test_edm::host input{m_host_mr};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create a (fixed sized) host buffer, to stage the data into.
    test_edm::buffer host_buffer1{
        static_cast<test_edm::buffer::size_type>(input.size()), m_host_mr,
        vecmem::data::buffer_type::resizable};
    m_host_copy.setup(host_buffer1);

    // Stage the data into the host buffer.
    m_host_copy(vecmem::get_data(input), host_buffer1);

    // Create the (fixed sized) device buffer.
    test_edm::buffer device_buffer{host_buffer1.capacity(), m_device_mr,
                                   vecmem::data::buffer_type::resizable};
    m_sycl_copy.setup(device_buffer);

    // Copy the data from the host buffer to the device buffer.
    m_sycl_copy(host_buffer1, device_buffer,
                vecmem::copy::type::host_to_device);

    // Create a (fixed sized) host buffer, to stage the data back into.
    test_edm::buffer host_buffer2{host_buffer1.capacity(), m_host_mr,
                                  vecmem::data::buffer_type::resizable};
    m_host_copy.setup(host_buffer2);

    // Copy the data from the device buffer to the host buffer.
    m_sycl_copy(device_buffer, host_buffer2,
                vecmem::copy::type::device_to_host);

    // Create the target host container.
    test_edm::host target{m_host_mr};

    // Copy the data from the host buffer to the target.
    m_host_copy(host_buffer2, target);

    // Compare the two.
    vecmem::testing::compare(input, target);
}
