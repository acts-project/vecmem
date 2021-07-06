/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/containers/data/jagged_vector_data.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <set>

class core_jagged_vector_view_test : public testing::Test {
    protected:
    vecmem::host_memory_resource m_mem;
    vecmem::jagged_vector<int> m_vec;
    vecmem::data::jagged_vector_data<int> m_data;
    vecmem::jagged_device_vector<int> m_jag;

    core_jagged_vector_view_test(void)
        : m_vec({vecmem::vector<int>({1, 2, 3, 4}, &m_mem),
                 vecmem::vector<int>({5, 6}, &m_mem),
                 vecmem::vector<int>({7, 8, 9, 10}, &m_mem),
                 vecmem::vector<int>({11}, &m_mem), vecmem::vector<int>(&m_mem),
                 vecmem::vector<int>({12, 13, 14, 15, 16}, &m_mem)},
                &m_mem),
          m_data(vecmem::get_data(m_vec)),
          m_jag(m_data) {}
};

TEST_F(core_jagged_vector_view_test, top_level_size) {
    EXPECT_EQ(m_jag.size(), 6);
}

TEST_F(core_jagged_vector_view_test, row_size) {
    EXPECT_EQ(m_jag.at(0).size(), 4);
    EXPECT_EQ(m_jag.at(1).size(), 2);
    EXPECT_EQ(m_jag.at(2).size(), 4);
    EXPECT_EQ(m_jag.at(3).size(), 1);
    EXPECT_EQ(m_jag.at(4).size(), 0);
    EXPECT_EQ(m_jag.at(5).size(), 5);
}

TEST_F(core_jagged_vector_view_test, two_d_access) {
    EXPECT_EQ(m_jag.at(0).at(0), 1);
    EXPECT_EQ(m_jag.at(0).at(1), 2);
    EXPECT_EQ(m_jag.at(0).at(2), 3);
    EXPECT_EQ(m_jag.at(0).at(3), 4);
    EXPECT_EQ(m_jag.at(1).at(0), 5);
    EXPECT_EQ(m_jag.at(1).at(1), 6);
    EXPECT_EQ(m_jag.at(2).at(0), 7);
    EXPECT_EQ(m_jag.at(2).at(1), 8);
    EXPECT_EQ(m_jag.at(2).at(2), 9);
    EXPECT_EQ(m_jag.at(2).at(3), 10);
}

TEST_F(core_jagged_vector_view_test, two_d_access_const) {
    const vecmem::jagged_device_vector<int>& jag = m_jag;
    EXPECT_EQ(jag.at(0).at(0), 1);
    EXPECT_EQ(jag.at(0).at(1), 2);
    EXPECT_EQ(jag.at(0).at(2), 3);
    EXPECT_EQ(jag.at(0).at(3), 4);
    EXPECT_EQ(jag.at(1).at(0), 5);
    EXPECT_EQ(jag.at(1).at(1), 6);
    EXPECT_EQ(jag.at(2).at(0), 7);
    EXPECT_EQ(jag.at(2).at(1), 8);
    EXPECT_EQ(jag.at(2).at(2), 9);
    EXPECT_EQ(jag.at(2).at(3), 10);
}

TEST_F(core_jagged_vector_view_test, mutate) {
    m_jag.at(0).at(0) *= 2;
    m_jag.at(0).at(1) *= 2;
    m_jag.at(0).at(2) *= 2;
    m_jag.at(0).at(3) *= 2;
    m_jag.at(1).at(0) *= 2;
    m_jag.at(1).at(1) *= 2;
    m_jag.at(2).at(0) *= 2;
    m_jag.at(2).at(1) *= 2;
    m_jag.at(2).at(2) *= 2;
    m_jag.at(2).at(3) *= 2;

    EXPECT_EQ(m_jag.at(0).at(0), 2 * 1);
    EXPECT_EQ(m_jag.at(0).at(1), 2 * 2);
    EXPECT_EQ(m_jag.at(0).at(2), 2 * 3);
    EXPECT_EQ(m_jag.at(0).at(3), 2 * 4);
    EXPECT_EQ(m_jag.at(1).at(0), 2 * 5);
    EXPECT_EQ(m_jag.at(1).at(1), 2 * 6);
    EXPECT_EQ(m_jag.at(2).at(0), 2 * 7);
    EXPECT_EQ(m_jag.at(2).at(1), 2 * 8);
    EXPECT_EQ(m_jag.at(2).at(2), 2 * 9);
    EXPECT_EQ(m_jag.at(2).at(3), 2 * 10);
}

TEST_F(core_jagged_vector_view_test, iterator) {
    std::size_t i = 0;
    for (auto itr = m_jag.begin(); itr != m_jag.end(); ++itr) {
        i += itr->size();
    }
    EXPECT_EQ(i, 16);
}

TEST_F(core_jagged_vector_view_test, reverse_iterator) {
    std::size_t i = 0;
    for (auto itr = m_jag.rbegin(); itr != m_jag.rend(); ++itr) {
        i += itr->size();
    }
    EXPECT_EQ(i, 16);
}

TEST_F(core_jagged_vector_view_test, value_iteration) {
    std::size_t i = 0;
    for (auto innerv : m_jag) {
        i += innerv.size();
    }
    EXPECT_EQ(i, 16);
}

TEST_F(core_jagged_vector_view_test, filter) {

    // Helper object for performing memory copies.
    vecmem::copy copy;

    // Create a resizable buffer for a jagged vector.
    vecmem::data::jagged_vector_buffer<int> output_data(
        {0, 0, 0, 0, 0, 0}, {10, 10, 10, 10, 10, 10}, m_mem);
    copy.setup(output_data);

    // Fill the jagged vector buffer with just the odd elements.
    vecmem::jagged_device_vector device_vec(output_data);
    for (std::size_t i = 0; i < m_vec.size(); ++i) {
        for (std::size_t j = 0; j < m_vec.at(i).size(); ++j) {
            if ((m_vec[i][j] % 2) != 0) {
                device_vec[i].push_back(m_vec[i][j]);
            }
        }
    }

    // Copy the filtered output back into a "host object".
    vecmem::jagged_vector<int> output(&m_mem);
    copy(output_data, output);

    // Check the output.
    EXPECT_EQ(output.size(), 6);
    EXPECT_EQ(std::set<int>(output[0].begin(), output[0].end()),
              std::set<int>({1, 3}));
    EXPECT_EQ(std::set<int>(output[1].begin(), output[1].end()),
              std::set<int>({5}));
    EXPECT_EQ(std::set<int>(output[2].begin(), output[2].end()),
              std::set<int>({7, 9}));
    EXPECT_EQ(std::set<int>(output[3].begin(), output[3].end()),
              std::set<int>({11}));
    EXPECT_EQ(std::set<int>(output[4].begin(), output[4].end()),
              std::set<int>({}));
    EXPECT_EQ(std::set<int>(output[5].begin(), output[5].end()),
              std::set<int>({13, 15}));
}
