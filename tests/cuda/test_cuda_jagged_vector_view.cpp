/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "test_cuda_jagged_vector_view_kernels.cuh"

#include "vecmem/memory/cuda/managed_memory_resource.hpp"

#include "vecmem/containers/array.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"

#include <gtest/gtest.h>

class cuda_jagged_vector_view_test : public testing::Test {
    protected:
    vecmem::cuda::managed_memory_resource m_mem;
    vecmem::jagged_vector<int> m_vec;
    vecmem::array< int, 2 > m_constants;

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
        m_constants( m_mem )
    {
        m_constants[ 0 ] = 2; m_constants[ 1 ] = 1;
    }
};

TEST_F(cuda_jagged_vector_view_test, mutate_in_kernel) {

    // Create the data object describing the jagged vector.
    auto vec_data = vecmem::get_data( m_vec );

    // Run the linear transformation.
    linearTransform( vecmem::get_data( m_constants ), vec_data, vec_data );

    // Check the results.
    EXPECT_EQ(m_vec.at(0).at(0), 214);
    EXPECT_EQ(m_vec.at(0).at(1), 5);
    EXPECT_EQ(m_vec.at(0).at(2), 7);
    EXPECT_EQ(m_vec.at(0).at(3), 9);
    EXPECT_EQ(m_vec.at(1).at(0), 222);
    EXPECT_EQ(m_vec.at(1).at(1), 13);
    EXPECT_EQ(m_vec.at(2).at(0), 226);
    EXPECT_EQ(m_vec.at(2).at(1), 17);
    EXPECT_EQ(m_vec.at(2).at(2), 19);
    EXPECT_EQ(m_vec.at(2).at(3), 21);
    EXPECT_EQ(m_vec.at(3).at(0), 234);
    EXPECT_EQ(m_vec.at(5).at(0), 236);
    EXPECT_EQ(m_vec.at(5).at(1), 27);
    EXPECT_EQ(m_vec.at(5).at(2), 29);
    EXPECT_EQ(m_vec.at(5).at(3), 31);
    EXPECT_EQ(m_vec.at(5).at(4), 33);
}
