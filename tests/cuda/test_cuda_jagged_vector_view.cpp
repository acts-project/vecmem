/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "test_cuda_jagged_vector_view_kernels.cuh"

#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/data/jagged_vector_data.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"

#include <gtest/gtest.h>

class cuda_jagged_vector_view_test : public testing::Test {
    protected:
    vecmem::cuda::managed_memory_resource m_mem;
    vecmem::jagged_vector<int> m_vec;
    vecmem::jagged_vector_data<int> m_data;

    cuda_jagged_vector_view_test(
        void
    ) :
        m_vec({
            vecmem::vector<int>({1, 2, 3, 4}, &m_mem),
            vecmem::vector<int>({5, 6}, &m_mem),
            vecmem::vector<int>({7, 8, 9, 10}, &m_mem),
            vecmem::vector<int>({11}, &m_mem),
            vecmem::vector<int>(&m_mem),
            vecmem::vector<int>({12, 13, 14, 15, 16}, &m_mem)
        }, &m_mem),
        m_data(m_vec, &m_mem)
    {
    }
};

TEST_F(cuda_jagged_vector_view_test, mutate_in_kernel) {
    doubleJagged(m_data);

    vecmem::jagged_device_vector<int> m_jag(m_data);

    EXPECT_EQ(m_jag.at(0, 0), 2);
    EXPECT_EQ(m_jag.at(0, 1), 4);
    EXPECT_EQ(m_jag.at(0, 2), 6);
    EXPECT_EQ(m_jag.at(0, 3), 8);
    EXPECT_EQ(m_jag.at(1, 0), 10);
    EXPECT_EQ(m_jag.at(1, 1), 12);
    EXPECT_EQ(m_jag.at(2, 0), 14);
    EXPECT_EQ(m_jag.at(2, 1), 16);
    EXPECT_EQ(m_jag.at(2, 2), 18);
    EXPECT_EQ(m_jag.at(2, 3), 20);
    EXPECT_EQ(m_jag.at(3, 0), 22);
    EXPECT_EQ(m_jag.at(5, 0), 24);
    EXPECT_EQ(m_jag.at(5, 1), 26);
    EXPECT_EQ(m_jag.at(5, 2), 28);
    EXPECT_EQ(m_jag.at(5, 3), 30);
    EXPECT_EQ(m_jag.at(5, 4), 32);
}
