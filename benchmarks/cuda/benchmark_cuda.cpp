/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

static vecmem::cuda::device_memory_resource device_mr;
void BenchmarkCudaDevice(benchmark::State& state) {
    for (auto _ : state) {
        void* p = device_mr.allocate(state.range(0));
        device_mr.deallocate(p, state.range(0));
    }
}
BENCHMARK(BenchmarkCudaDevice)->RangeMultiplier(2)->Range(1, 2UL << 31);

static vecmem::cuda::host_memory_resource host_mr;
void BenchmarkCudaPinned(benchmark::State& state) {
    for (auto _ : state) {
        void* p = host_mr.allocate(state.range(0));
        host_mr.deallocate(p, state.range(0));
    }
}
BENCHMARK(BenchmarkCudaPinned)->RangeMultiplier(2)->Range(1, 2UL << 31);

static vecmem::cuda::managed_memory_resource managed_mr;
void BenchmarkCudaManaged(benchmark::State& state) {
    for (auto _ : state) {
        void* p = managed_mr.allocate(state.range(0));
        managed_mr.deallocate(p, state.range(0));
    }
}
BENCHMARK(BenchmarkCudaManaged)->RangeMultiplier(2)->Range(1, 2UL << 31);
