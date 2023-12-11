/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/buffer.hpp"
#include "vecmem/edm/device.hpp"
#include "vecmem/edm/host.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test case for the EDM code
class core_edm_copy_test : public testing::Test {

protected:
    /// Schema without any jagged vectors.
    using simple_schema = vecmem::edm::schema<
        vecmem::edm::type::scalar<int>, vecmem::edm::type::vector<float>,
        vecmem::edm::type::scalar<double>, vecmem::edm::type::vector<double>,
        vecmem::edm::type::vector<int>>;
    /// Constant schema without any jagged vectors.
    using simple_const_schema =
        vecmem::edm::details::add_const_t<simple_schema>;

    /// Schema with some jagged vectors.
    using jagged_schema =
        vecmem::edm::schema<vecmem::edm::type::vector<float>,
                            vecmem::edm::type::jagged_vector<double>,
                            vecmem::edm::type::scalar<int>,
                            vecmem::edm::type::jagged_vector<int>>;
    /// Constant schema with some jagged vectors.
    using jagged_const_schema =
        vecmem::edm::details::add_const_t<jagged_schema>;

    /// Fill a host container with some data.
    void fill(vecmem::edm::host<simple_schema>& c) const {
        c.resize(3);
        c.get<0>() = 1;
        c.get<1>() = {2.f, 3.f, 4.f};
        c.get<2>() = 5.;
        c.get<3>() = {6., 7., 8.};
        c.get<4>() = {9, 10, 11};
    }
    /// Fill a host container with some data.
    void fill(vecmem::edm::host<jagged_schema>& c) const {
        c.resize(3);
        c.get<0>() = {1.f, 2.f, 3.f};
        c.get<1>()[0] = {4., 5.};
        c.get<1>()[1] = {};
        c.get<1>()[2] = {6., 7., 8.};
        c.get<2>() = 9;
        c.get<3>()[0] = {10, 11};
        c.get<3>()[1] = {};
        c.get<3>()[2] = {12, 13, 14};
    }
    /// Compare two containers.
    void compare(const vecmem::edm::view<simple_const_schema>& c1,
                 const vecmem::edm::view<simple_const_schema>& c2) const {
        // Create device containers.
        const vecmem::edm::device<simple_const_schema> d1{c1};
        const vecmem::edm::device<simple_const_schema> d2{c2};
        // Compare them.
        EXPECT_EQ(d1.get<0>(), d2.get<0>());
        ASSERT_EQ(d1.get<1>().size(), d2.get<1>().size());
        for (unsigned int i = 0; i < d1.get<1>().size(); ++i) {
            EXPECT_FLOAT_EQ(d1.get<1>()[i], d2.get<1>()[i]);
        }
        EXPECT_DOUBLE_EQ(d1.get<2>(), d2.get<2>());
        ASSERT_EQ(d1.get<3>().size(), d2.get<3>().size());
        for (unsigned int i = 0; i < d1.get<3>().size(); ++i) {
            EXPECT_DOUBLE_EQ(d1.get<3>()[i], d2.get<3>()[i]);
        }
        ASSERT_EQ(d1.get<4>().size(), d2.get<4>().size());
        for (unsigned int i = 0; i < d1.get<4>().size(); ++i) {
            EXPECT_EQ(d1.get<4>()[i], d2.get<4>()[i]);
        }
    }
    /// Compare two containers.
    void compare(const vecmem::edm::view<jagged_const_schema>& c1,
                 const vecmem::edm::view<jagged_const_schema>& c2) const {
        // Create device containers.
        const vecmem::edm::device<jagged_const_schema> d1{c1};
        const vecmem::edm::device<jagged_const_schema> d2{c2};
        // Compare them.
        ASSERT_EQ(d1.get<0>().size(), d2.get<0>().size());
        for (unsigned int i = 0; i < d1.get<0>().size(); ++i) {
            EXPECT_FLOAT_EQ(d1.get<0>()[i], d2.get<0>()[i]);
        }
        ASSERT_EQ(d1.get<1>().size(), d2.get<1>().size());
        for (unsigned int i = 0; i < d1.get<1>().size(); ++i) {
            ASSERT_EQ(d1.get<1>()[i].size(), d2.get<1>()[i].size());
            for (unsigned int j = 0; j < d1.get<1>()[i].size(); ++j) {
                EXPECT_DOUBLE_EQ(d1.get<1>()[i][j], d2.get<1>()[i][j]);
            }
        }
        EXPECT_EQ(d1.get<2>(), d2.get<2>());
        ASSERT_EQ(d1.get<3>().size(), d2.get<3>().size());
        for (unsigned int i = 0; i < d1.get<3>().size(); ++i) {
            ASSERT_EQ(d1.get<3>()[i].size(), d2.get<3>()[i].size());
            for (unsigned int j = 0; j < d1.get<3>()[i].size(); ++j) {
                EXPECT_EQ(d1.get<3>()[i][j], d2.get<3>()[i][j]);
            }
        }
    }

    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;
    /// Helper object for the memory copies.
    vecmem::copy m_copy;

};  // class core_edm_copy_test

TEST_F(core_edm_copy_test, host_to_host_simple) {

    vecmem::edm::host<simple_schema> c1{m_resource};
    vecmem::edm::host<simple_schema> c2{m_resource};

    fill(c1);
    m_copy(vecmem::get_data(c1), c2);
    compare(vecmem::get_data(c1), vecmem::get_data(c2));
}

TEST_F(core_edm_copy_test, host_to_host_jagged) {

    vecmem::edm::host<jagged_schema> c1{m_resource};
    vecmem::edm::host<jagged_schema> c2{m_resource};

    fill(c1);
    m_copy(vecmem::get_data(c1), c2, vecmem::copy::type::host_to_host);
    compare(vecmem::get_data(c1), vecmem::get_data(c2));
}

TEST_F(core_edm_copy_test, host_to_fixed_device_simple) {

    vecmem::edm::host<simple_schema> host{m_resource};
    fill(host);

    // Test that a copy into a too small buffer would fail correctly.
    vecmem::edm::buffer<simple_schema> buffer1{
        2u, m_resource, vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer1);
    EXPECT_THROW(m_copy(vecmem::get_data(host), buffer1,
                        vecmem::copy::type::host_to_host),
                 std::length_error);

    // Test a valid copy.
    vecmem::edm::buffer<simple_schema> buffer2{
        static_cast<vecmem::edm::buffer<jagged_schema>::size_type>(host.size()),
        m_resource, vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer2);
    m_copy(vecmem::get_data(host), buffer2, vecmem::copy::type::host_to_host);

    compare(vecmem::get_data(host), vecmem::get_data(buffer2));
}

TEST_F(core_edm_copy_test, host_to_fixed_device_jagged) {

    vecmem::edm::host<jagged_schema> host{m_resource};
    fill(host);

    // Test that a copy into a too small buffer would fail correctly.
    vecmem::edm::buffer<jagged_schema> buffer1{
        std::vector<std::size_t>{2, 0, 2}, m_resource, nullptr,
        vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer1);
    EXPECT_THROW(m_copy(vecmem::get_data(host), buffer1,
                        vecmem::copy::type::host_to_host),
                 std::length_error);

    // Test a valid copy.
    vecmem::edm::buffer<jagged_schema> buffer2{
        std::vector<std::size_t>{2, 0, 3}, m_resource, nullptr,
        vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer2);
    m_copy(vecmem::get_data(host), buffer2, vecmem::copy::type::host_to_host);

    compare(vecmem::get_data(host), vecmem::get_data(buffer2));
}

TEST_F(core_edm_copy_test, host_to_resizable_device_simple) {

    vecmem::edm::host<simple_schema> host{m_resource};
    fill(host);

    vecmem::edm::buffer<simple_schema> buffer{
        10u, m_resource, vecmem::data::buffer_type::resizable};
    m_copy.setup(buffer);

    m_copy(vecmem::get_data(host), buffer, vecmem::copy::type::host_to_host);

    compare(vecmem::get_data(host), vecmem::get_data(buffer));
}

TEST_F(core_edm_copy_test, host_to_resizable_device_jagged) {

    vecmem::edm::host<jagged_schema> host{m_resource};
    fill(host);

    vecmem::edm::buffer<jagged_schema> buffer{
        std::vector<std::size_t>{5, 5, 5}, m_resource, nullptr,
        vecmem::data::buffer_type::resizable};
    m_copy.setup(buffer);

    m_copy(vecmem::get_data(host), buffer, vecmem::copy::type::host_to_host);

    compare(vecmem::get_data(host), vecmem::get_data(buffer));
}

TEST_F(core_edm_copy_test, host_to_device_to_fixed_device_simple) {

    vecmem::edm::host<simple_schema> host{m_resource};
    fill(host);

    vecmem::edm::buffer<simple_schema> buffer1{
        static_cast<vecmem::edm::buffer<jagged_schema>::size_type>(host.size()),
        m_resource, vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer1);

    vecmem::edm::buffer<simple_schema> buffer2{
        static_cast<vecmem::edm::buffer<jagged_schema>::size_type>(host.size()),
        m_resource, vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer2);

    m_copy(vecmem::get_data(host), buffer1, vecmem::copy::type::host_to_host);
    m_copy(buffer1, buffer2, vecmem::copy::type::host_to_host);

    compare(vecmem::get_data(host), vecmem::get_data(buffer2));
}

TEST_F(core_edm_copy_test, host_to_device_to_fixed_device_jagged) {

    vecmem::edm::host<jagged_schema> host{m_resource};
    fill(host);

    vecmem::edm::buffer<jagged_schema> buffer1{
        std::vector<std::size_t>{2, 0, 3}, m_resource, nullptr,
        vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer1);

    vecmem::edm::buffer<jagged_schema> buffer2{
        std::vector<std::size_t>{2, 0, 3}, m_resource, nullptr,
        vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer2);

    m_copy(vecmem::get_data(host), buffer1, vecmem::copy::type::host_to_host);
    m_copy(buffer1, buffer2, vecmem::copy::type::host_to_host);

    compare(vecmem::get_data(host), vecmem::get_data(buffer2));
}

TEST_F(core_edm_copy_test, host_to_device_to_resizable_device_simple) {

    vecmem::edm::host<simple_schema> host{m_resource};
    fill(host);

    vecmem::edm::buffer<simple_schema> buffer1{
        static_cast<vecmem::edm::buffer<jagged_schema>::size_type>(host.size()),
        m_resource, vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer1);

    vecmem::edm::buffer<simple_schema> buffer2{
        10u, m_resource, vecmem::data::buffer_type::resizable};
    m_copy.setup(buffer2);

    m_copy(vecmem::get_data(host), buffer1, vecmem::copy::type::host_to_host);
    m_copy(buffer1, buffer2, vecmem::copy::type::host_to_host);

    compare(vecmem::get_data(host), vecmem::get_data(buffer2));
}

TEST_F(core_edm_copy_test, host_to_device_to_resizable_device_jagged) {

    vecmem::edm::host<jagged_schema> host{m_resource};
    fill(host);

    vecmem::edm::buffer<jagged_schema> buffer1{
        std::vector<std::size_t>{2, 0, 3}, m_resource, nullptr,
        vecmem::data::buffer_type::fixed_size};
    m_copy.setup(buffer1);

    vecmem::edm::buffer<jagged_schema> buffer2{
        std::vector<std::size_t>{5, 5, 5}, m_resource, nullptr,
        vecmem::data::buffer_type::resizable};
    m_copy.setup(buffer2);

    m_copy(vecmem::get_data(host), buffer1, vecmem::copy::type::host_to_host);
    m_copy(buffer1, buffer2, vecmem::copy::type::host_to_host);

    compare(vecmem::get_data(host), vecmem::get_data(buffer2));
}
