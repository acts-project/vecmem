/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/device.hpp"
#include "vecmem/edm/host.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test case for @c vecmem::edm::host.
class core_edm_host_test : public testing::Test {

protected:
    /// Schema for the tests.
    using schema =
        vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                            vecmem::edm::type::vector<float>,
                            vecmem::edm::type::jagged_vector<double>>;
    /// Constant schema used for the test.
    using const_schema = vecmem::edm::details::add_const_t<schema>;

    /// Create a host container object with some non-trivial content.
    vecmem::edm::host<schema> create() {

        // Create a host container, and fill it.
        vecmem::edm::host<schema> host{m_resource};
        static constexpr std::size_t SIZE = 5;
        host.resize(SIZE);
        host.get<0>() = 1;
        for (std::size_t i = 0; i < SIZE; ++i) {
            host.get<1>()[i] = static_cast<float>(2 + i);
            host.get<2>()[i].resize(i + 1);
            for (std::size_t j = 0; j < i + 1; ++j) {
                host.get<2>()[i][j] = static_cast<double>(3 + i + j);
            }
        }
        // Give it to the caller.
        return host;
    }

    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;

};  // class core_edm_host_test

TEST_F(core_edm_host_test, construct_assign) {

    // Construct a host container, and make sure that it looks okay.
    vecmem::edm::host<schema> host1 = create();
    EXPECT_EQ(host1.size(), host1.get<1>().size());
    EXPECT_EQ(host1.size(), host1.get<2>().size());

    // Lambda comparing two host containers.
    auto compare = [](const vecmem::edm::host<schema>& h1,
                      const vecmem::edm::host<schema>& h2) {
        EXPECT_EQ(h1.size(), h2.size());
        EXPECT_EQ(h1.get<0>(), h2.get<0>());
        EXPECT_EQ(h1.get<1>(), h2.get<1>());
        EXPECT_EQ(h1.get<2>(), h2.get<2>());
    };

    // Construct a copy of it, and check that it is the same.
    vecmem::edm::host<schema> host2{host1};
    compare(host1, host2);

    // Create a new host container, and assign the first one to it. Check that
    // this also works as it should.
    vecmem::edm::host<schema> host3{m_resource};
    host3 = host1;
    compare(host1, host3);
}

TEST_F(core_edm_host_test, get_data) {

    // Construct a host container.
    vecmem::edm::host<schema> host1 = create();

    // Lambda checking a data object against a host container.
    auto compare = [](const auto& data, const vecmem::edm::host<schema>& host) {
        EXPECT_EQ(data.capacity(), host.size());
        EXPECT_EQ(data.template get<0>(), &(host.get<0>()));
        EXPECT_EQ(data.template get<1>().size(), host.size());
        EXPECT_EQ(data.template get<1>().capacity(), host.size());
        EXPECT_EQ(data.template get<1>().ptr(), host.get<1>().data());
        ASSERT_EQ(data.template get<2>().size(), host.size());
        EXPECT_EQ(data.template get<2>().capacity(), host.size());
        for (std::size_t i = 0; i < host.size(); ++i) {
            EXPECT_EQ(data.template get<2>().host_ptr(),
                      data.template get<2>().ptr());
            EXPECT_EQ(data.template get<2>().host_ptr()[i].size(),
                      host.get<2>()[i].size());
            EXPECT_EQ(data.template get<2>().host_ptr()[i].capacity(),
                      host.get<2>()[i].size());
            EXPECT_EQ(data.template get<2>().host_ptr()[i].ptr(),
                      host.get<2>()[i].data());
        }
    };

    // Get a non-const data object for it, and check its contents.
    vecmem::edm::data<schema> data1 = vecmem::get_data(host1, &m_resource);
    compare(data1, host1);

    // Get a const data object for it, and check its contents.
    vecmem::edm::data<const_schema> data2 =
        [](const vecmem::edm::host<schema>& host) {
            return vecmem::get_data(host);
        }(host1);
    compare(data2, host1);
}

TEST_F(core_edm_host_test, device) {

    // Construct a host container.
    vecmem::edm::host<schema> host1 = create();

    // Lambda comparing the contents of a host and a device container.
    auto compare = [](const auto& device,
                      const vecmem::edm::host<schema>& host) {
        ASSERT_EQ(device.size(), host.size());
        EXPECT_EQ(device.template get<0>(), host.get<0>());
        for (vecmem::edm::details::size_type i = 0; i < device.size(); ++i) {
            EXPECT_EQ(device.template get<1>()[i], host.get<1>()[i]);
            ASSERT_EQ(device.template get<2>()[i].size(),
                      host.get<2>()[i].size());
            for (vecmem::edm::details::size_type j = 0;
                 j < device.template get<2>()[i].size(); ++j) {
                EXPECT_EQ(device.template get<2>()[i][j], host.get<2>()[i][j]);
            }
        }
    };

    // Create a non-const device object for it, and check its contents.
    auto data1 = vecmem::get_data(host1);
    vecmem::edm::device<schema> device1{data1};
    compare(device1, host1);

    // Create constant device objects for it, and check their contents.
    auto data2 = [](const vecmem::edm::host<schema>& host) {
        return vecmem::get_data(host);
    }(host1);
    vecmem::edm::device<const_schema> device2{data2};
    compare(device2, host1);
    vecmem::edm::device<const_schema> device3{data1};
    compare(device3, host1);
}
