/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/edm_simple_container.hpp"
#include "../common/edm_simple_container_helpers.hpp"
#include "test_hip_edm_kernels.hpp"
#include "vecmem/memory/hip/device_memory_resource.hpp"
#include "vecmem/memory/hip/host_memory_resource.hpp"
#include "vecmem/utils/hip/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test fixture for the HIP EDM tests
class hip_edm_test : public testing::Test {

protected:
    /// Host memory resource
    vecmem::hip::host_memory_resource m_host_mr;
    /// Device memory resource
    vecmem::hip::device_memory_resource m_device_mr;

    /// Helper object for performing (HIP) memory copies
    vecmem::hip::copy m_copy;

};  // class hip_edm_test

/// Convenience type declaration
using test_edm = vecmem::testing::simple_container;

TEST_F(hip_edm_test, modify_host) {

    // Create two host containers.
    test_edm::host container1{m_host_mr}, container2{m_host_mr};

    // Fill them with some data.
    vecmem::testing::fill(container1);
    vecmem::testing::fill(container2);

    // Modify the first container, using a simple for loop.
    test_edm::device device1 = vecmem::get_data(container1);
    for (unsigned int i = 0; i < container1.size(); ++i) {
        modify(i, device1);
    }

    // Run a kernel that executes the modify function on the second container.
    edmModify(vecmem::get_data(container2));

    // Compare the two.
    vecmem::testing::compare(container1, container2);
}

TEST_F(hip_edm_test, modify_device) {

    // Create a host container in host memory as a start.
    test_edm::host container1{m_host_mr};

    // Fill it with some data.
    vecmem::testing::fill(container1);

    // Copy it to the device.
    test_edm::buffer buffer{
        static_cast<test_edm::buffer::size_type>(container1.size()),
        m_device_mr};
    m_copy.setup(buffer);
    m_copy(vecmem::get_data(container1), buffer,
           vecmem::copy::type::host_to_device);

    // Modify the container in host memory, using a simple for loop.
    test_edm::device device1 = vecmem::get_data(container1);
    for (unsigned int i = 0; i < container1.size(); ++i) {
        modify(i, device1);
    }

    // Run a kernel that executes the modify function on the device buffer.
    edmModify(buffer);

    // Copy the data back to the host.
    test_edm::host container2{m_host_mr};
    m_copy(buffer, container2);

    // Compare the two.
    vecmem::testing::compare(container1, container2);
}

TEST_F(hip_edm_test, fill_device) {

    // Container sizes to create on the host and the device.
    static constexpr unsigned int CONTAINER_SIZE = 10000;

    // Create a host container, and fill it.
    test_edm::host container1{m_host_mr};
    container1.resize(CONTAINER_SIZE);
    test_edm::device device1{vecmem::get_data(container1)};
    for (unsigned int i = 0; i < container1.size(); ++i) {
        fill(i, device1);
    }

    // Create a resizable device buffer.
    test_edm::buffer buffer{CONTAINER_SIZE, m_device_mr,
                            vecmem::data::buffer_type::resizable};
    m_copy.setup(buffer);

    // Run a kernel that fills the buffer.
    edmFill(buffer);

    // Copy the device buffer back to the host.
    test_edm::host container2{m_host_mr};
    m_copy(buffer, container2);

    // Compare the two containers.
    vecmem::testing::compare(container1, container2);
}
