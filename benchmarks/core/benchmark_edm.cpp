/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include <vecmem/edm/container.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

namespace vecmem::benchmark {

/// @name SoA to use in the benchmarking
/// @{

template <typename BASE>
struct TestSoA : public BASE {
    using BASE::BASE;
    using BASE::operator=;
};

using TestSoAContainer = vecmem::edm::container<
    TestSoA, vecmem::edm::type::scalar<float>, vecmem::edm::type::vector<float>,
    vecmem::edm::type::vector<double>, vecmem::edm::type::vector<float>,
    vecmem::edm::type::jagged_vector<float>>;

/// @}

/// The (host) memory resource to use in the benchmark(s)
static vecmem::host_memory_resource host_mr;

void edmHostPushBack(::benchmark::State& state) {

    const std::size_t size = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        TestSoAContainer::host soa{host_mr};
        for (std::size_t i = 0; i < size; ++i) {
            const float multiplier = static_cast<float>(i);
            soa.push_back({1.f,
                           multiplier * 1.f,
                           multiplier * 2.,
                           multiplier * 3.f,
                           {multiplier * 4.f, multiplier * 5.f}});
        }
    }
}
BENCHMARK(edmHostPushBack)->RangeMultiplier(2)->Range(1, 2UL << 15);

void edmHostReservedPushBack(::benchmark::State& state) {

    const std::size_t size = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        TestSoAContainer::host soa{host_mr};
        soa.reserve(size);
        for (std::size_t i = 0; i < size; ++i) {
            const float multiplier = static_cast<float>(i);
            soa.push_back({1.f,
                           multiplier * 1.f,
                           multiplier * 2.,
                           multiplier * 3.f,
                           {multiplier * 4.f, multiplier * 5.f}});
        }
    }
}
BENCHMARK(edmHostReservedPushBack)->RangeMultiplier(2)->Range(1, 2UL << 15);

}  // namespace vecmem::benchmark
