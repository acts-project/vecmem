/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/container.hpp"
#include "vecmem/utils/types.hpp"

namespace vecmem {
namespace benchmark {

/// Interface to a "simple container" used for benchmarking.
template <typename BASE>
class simple_soa : public BASE {

public:
    /// Inherit the base class's constructor(s)
    using BASE::BASE;

    /// "Count" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& count() { return BASE::template get<0>(); }
    /// "Count" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& count() const { return BASE::template get<0>(); }

    /// "Measurement" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& measurement() { return BASE::template get<1>(); }
    /// "Measurement" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& measurement() const { return BASE::template get<1>(); }

    /// "Average" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& average() { return BASE::template get<2>(); }
    /// "Average" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& average() const { return BASE::template get<2>(); }

    /// "Index" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& index() { return BASE::template get<3>(); }
    /// "Index" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& index() const { return BASE::template get<3>(); }

};  // class simple_soa

/// "Simple" container for the tests
///
/// Meaning that it would not have any jagged vector variables in it...
///
using simple_soa_container =
    edm::container<simple_soa, edm::type::vector<int>, edm::type::vector<float>,
                   edm::type::vector<float>, edm::type::vector<int> >;

}  // namespace benchmark
}  // namespace vecmem
