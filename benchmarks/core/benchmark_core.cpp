#include <benchmark/benchmark.h>

#include <vecmem/memory/host_memory_resource.hpp>

#include "../common/benchmark_memory_resource.hpp"

static void BenchmarkHost(benchmark::State& state) {
    vecmem::host_memory_resource r;

    for (auto _ : state) {
        void* p = r.allocate(state.range(0));
        r.deallocate(p, state.range(0));
    }
}

BENCHMARK(BenchmarkHost)->RangeMultiplier(2)->Range(1, 2UL << 31);
