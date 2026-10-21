/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/vector.hpp"

namespace vecmem {
namespace benchmark {

/// Simple AoS struct used for benchmarking.
struct simple_aos {

    int count;
    float measurement;
    float average;
    int index;

};  // class simple_aos

/// "Simple" container for the benchmarks
struct simple_aos_container {

#if __cplusplus >= 201700L
    /// Host container
    using host = vector<simple_aos>;
    /// Host buffer
    using buffer = data::vector_buffer<simple_aos>;
#endif  // __cplusplus >= 201700L

    /// Non-const device collection for @c simple_aos
    using device = device_vector<simple_aos>;
    /// Constant device collection for @c simple_aos
    using const_device = device_vector<const simple_aos>;

    /// Non-constant view of an @c simple_aos collection
    using view = data::vector_view<simple_aos>;
    /// Constant view of an @c simple_aos collection
    using const_view = data::vector_view<const simple_aos>;

};  // struct simple_aos_container

}  // namespace benchmark
}  // namespace vecmem
