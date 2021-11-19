#include <benchmark/benchmark.h>

#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>

#include "../common/benchmark_memory_resource.hpp"

static void BenchmarkCudaDevice(benchmark::State& state) {
    vecmem::cuda::device_memory_resource r;

    for (auto _ : state) {
        void* p = r.allocate(state.range(0));
        r.deallocate(p, state.range(0));
    }
}

static void BenchmarkCudaPinned(benchmark::State& state) {
    vecmem::cuda::host_memory_resource r;

    for (auto _ : state) {
        void* p = r.allocate(state.range(0));
        r.deallocate(p, state.range(0));
    }
}

static void BenchmarkCudaManaged(benchmark::State& state) {
    vecmem::cuda::managed_memory_resource r;

    for (auto _ : state) {
        void* p = r.allocate(state.range(0));
        r.deallocate(p, state.range(0));
    }
}

BENCHMARK(BenchmarkCudaDevice)->RangeMultiplier(2)->Range(1, 2UL << 31);

BENCHMARK(BenchmarkCudaPinned)->RangeMultiplier(2)->Range(1, 2UL << 31);

BENCHMARK(BenchmarkCudaManaged)->RangeMultiplier(2)->Range(1, 2UL << 31);
