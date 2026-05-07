/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Google include(s).
#include <benchmark/benchmark.h>

namespace vecmem::benchmark {

/// Set up the custom "counters" for the "simple copy" benchmarks.
///
/// @param state The benchmark state to set up the counters for.
/// @return The size of the benchmarked container.
///
std::size_t setup_simple_copy_counters(::benchmark::State& state);

}  // namespace vecmem::benchmark
