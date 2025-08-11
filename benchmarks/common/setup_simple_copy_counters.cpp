/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "setup_simple_copy_counters.hpp"

namespace vecmem::benchmark {

/// Set up the custom "counters" for the "simple copy" benchmarks.
std::size_t setup_simple_copy_counters(::benchmark::State& state) {

    // Get the size of the host container to create.
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    static constexpr std::size_t element_size =
        2 * sizeof(int) + 2 * sizeof(float);
    const double bytes = static_cast<double>(size * element_size);

    // Set custom "counters" for the benchmark.
    state.counters["Bytes"] = ::benchmark::Counter(
        bytes, ::benchmark::Counter::kDefaults, ::benchmark::Counter::kIs1024);
    state.counters["Rate"] = ::benchmark::Counter(
        bytes, ::benchmark::Counter::kIsIterationInvariantRate,
        ::benchmark::Counter::kIs1024);

    // Return the size of the benchmarked container.
    return size;
}

}  // namespace vecmem::benchmark
