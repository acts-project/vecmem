/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/memory/hip/managed_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

#include <vector>

static vecmem::hip::device_memory_resource device_mr;
void BenchmarkHipDevice(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        void* p = device_mr.allocate(size);
        device_mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkHipDevice)->RangeMultiplier(2)->Range(1, 2UL << 31);

void BenchmarkHipDeviceBinaryPage(benchmark::State& state) {
    std::size_t size = static_cast<std::size_t>(state.range(0));

    vecmem::binary_page_memory_resource mr(device_mr);

    for (auto _ : state) {
        void* p = mr.allocate(size);
        mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkHipDeviceBinaryPage)
    ->RangeMultiplier(2)
    ->Range(1, 2UL << 31);

void BenchmarkHipDeviceBinaryPageMultiple(benchmark::State& state) {
    std::size_t size = static_cast<std::size_t>(state.range(0));
    std::size_t nallocs = static_cast<std::size_t>(state.range(1));

    std::vector<void*> allocs;

    allocs.reserve(nallocs);

    for (auto _ : state) {
        vecmem::binary_page_memory_resource mr(device_mr);
        for (std::size_t i = 0; i < nallocs; ++i) {
            allocs[i] = mr.allocate(size);
        }

        for (std::size_t i = 0; i < nallocs; ++i) {
            mr.deallocate(allocs[i], size);
        }
    }
}
BENCHMARK(BenchmarkHipDeviceBinaryPageMultiple)
    ->RangeMultiplier(2)
    ->Ranges({{1, 2UL << 21}, {1, 1024}});

static vecmem::hip::host_memory_resource host_mr;
void BenchmarkHipPinned(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        void* p = host_mr.allocate(size);
        host_mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkHipPinned)->RangeMultiplier(2)->Range(1, 2UL << 31);

void BenchmarkHipPinnedBinaryPage(benchmark::State& state) {
    std::size_t size = static_cast<std::size_t>(state.range(0));

    vecmem::binary_page_memory_resource mr(host_mr);

    for (auto _ : state) {
        void* p = mr.allocate(size);
        mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkHipPinnedBinaryPage)
    ->RangeMultiplier(2)
    ->Range(1, 2UL << 31);

static vecmem::hip::managed_memory_resource managed_mr;
void BenchmarkHipManaged(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        void* p = managed_mr.allocate(size);
        managed_mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkHipManaged)->RangeMultiplier(2)->Range(1, 2UL << 31);

void BenchmarkHipManagedBinaryPage(benchmark::State& state) {
    std::size_t size = static_cast<std::size_t>(state.range(0));

    vecmem::binary_page_memory_resource mr(managed_mr);

    for (auto _ : state) {
        void* p = mr.allocate(size);
        mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkHipManagedBinaryPage)
    ->RangeMultiplier(2)
    ->Range(1, 2UL << 31);
