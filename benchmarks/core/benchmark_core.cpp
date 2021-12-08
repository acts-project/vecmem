/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

/// The (host) memory resource to use in the benchmark(s)
static vecmem::host_memory_resource host_mr;

void BenchmarkHost(benchmark::State& state) {
    for (auto _ : state) {
        void* p = host_mr.allocate(state.range(0));
        host_mr.deallocate(p, state.range(0));
    }
}

BENCHMARK(BenchmarkHost)->RangeMultiplier(2)->Range(1, 2UL << 31);
