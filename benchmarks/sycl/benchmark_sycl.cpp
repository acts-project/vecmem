/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/memory/sycl/shared_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

static vecmem::sycl::device_memory_resource device_mr;
void BenchmarkSYCLDevice(benchmark::State& state) {
    for (auto _ : state) {
        void* p = device_mr.allocate(state.range(0));
        device_mr.deallocate(p, state.range(0));
    }
}
BENCHMARK(BenchmarkSYCLDevice)->RangeMultiplier(2)->Range(1, 2UL << 31);

static vecmem::sycl::host_memory_resource host_mr;
void BenchmarkSYCLHost(benchmark::State& state) {
    for (auto _ : state) {
        void* p = host_mr.allocate(state.range(0));
        host_mr.deallocate(p, state.range(0));
    }
}
BENCHMARK(BenchmarkSYCLHost)->RangeMultiplier(2)->Range(1, 2UL << 31);

static vecmem::sycl::shared_memory_resource shared_mr;
void BenchmarkSYCLShared(benchmark::State& state) {
    for (auto _ : state) {
        void* p = shared_mr.allocate(state.range(0));
        shared_mr.deallocate(p, state.range(0));
    }
}
BENCHMARK(BenchmarkSYCLShared)->RangeMultiplier(2)->Range(1, 2UL << 31);
