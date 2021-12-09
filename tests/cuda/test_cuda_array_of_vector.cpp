/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "test_cuda_containers_kernels.cuh"
#include "vecmem/containers/array.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/cuda/async_copy.hpp"
#include "vecmem/utils/cuda/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test fixture for the on-device vecmem container tests
class cuda_array_of_vector_test : public testing::Test {

};  // class cuda_containers_test

template <typename T, std::size_t N, std::size_t index_t = N, typename... Ts>
auto make_array_with_args(T t, Ts... ts) {
    if constexpr (index_t <= 1) {
        return vecmem::static_array<T, N>{t, ts...};
    } else {
        return make_array_with_args<T, N, index_t - 1>(t, t, ts...);
    }
}

template <typename T, std::size_t N, std::size_t... ints>
auto make_data_array(vecmem::static_array<vecmem::vector<T>, N>& arr,
                     std::index_sequence<ints...> seq) {
    return vecmem::static_array<vecmem::data::vector_view<T>, N>{
        vecmem::get_data(arr[ints])...};
}

template <typename T, std::size_t N>
auto make_data_array(vecmem::static_array<vecmem::vector<T>, N>& arr) {
    auto seq = std::make_index_sequence<N>{};
    return make_data_array(arr, seq);
}

TEST_F(cuda_array_of_vector_test, array_of_vector) {

    // The managed memory resource.
    vecmem::cuda::managed_memory_resource mng_mr;

    auto arr_vec = make_array_with_args<vecmem::vector<int>, 3>(
        vecmem::vector<int>{&mng_mr});

    arr_vec[0] = vecmem::vector<int>({1, 9, 3}, &mng_mr);
    arr_vec[1] = vecmem::vector<int>({6, 8}, &mng_mr);
    arr_vec[2] = vecmem::vector<int>({0, 2}, &mng_mr);

    auto arr_data = make_data_array(arr_vec);

    readArray(arr_data);
}
