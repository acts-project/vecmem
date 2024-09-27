/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// Common benchmark include(s).
#include "../common/simple_aos_container.hpp"
#include "../common/simple_soa_container.hpp"

// Google benchmark include(s).
#include <benchmark/benchmark.h>

namespace vecmem::benchmark {

/// The (host) memory resource to use in the benchmark(s).
static host_memory_resource host_mr;
/// The copy object to use in the benchmark(s).
static copy host_copy;

void simpleSoADirectHostToFixedBufferCopy(::benchmark::State& state) {

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

    // Create the source host container.
    simple_soa_container::host source{host_mr};
    source.resize(size);

    // Create the destination (host) buffer.
    simple_soa_container::buffer dest{
        static_cast<simple_soa_container::buffer::size_type>(size), host_mr};
    host_copy.setup(dest)->ignore();

    // Perform the copy benchmark.
    for (auto _ : state) {
        host_copy(get_data(source), dest)->ignore();
    }
}
BENCHMARK(simpleSoADirectHostToFixedBufferCopy)->Range(1UL, 1UL << 26);

void simpleSoAOptimalHostToFixedBufferCopy(::benchmark::State& state) {

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

    // Create the source host container.
    simple_soa_container::host host{host_mr};
    host.resize(size);
    simple_soa_container::buffer source{
        static_cast<simple_soa_container::buffer::size_type>(size), host_mr};
    host_copy.setup(source)->ignore();
    host_copy(get_data(host), source)->ignore();

    // Create the destination (host) buffer.
    simple_soa_container::buffer dest{
        static_cast<simple_soa_container::buffer::size_type>(size), host_mr};
    host_copy.setup(dest)->ignore();

    // Perform the copy benchmark.
    for (auto _ : state) {
        host_copy(source, dest)->ignore();
    }
}
BENCHMARK(simpleSoAOptimalHostToFixedBufferCopy)->Range(1UL, 1UL << 26);

void simpleAoSHostToFixedBufferCopy(::benchmark::State& state) {

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

    // Create the source host container.
    simple_aos_container::host source{&host_mr};
    source.resize(size);

    // Create the destination (host) buffer.
    simple_aos_container::buffer dest{
        static_cast<simple_aos_container::buffer::size_type>(size), host_mr};
    host_copy.setup(dest)->ignore();

    // Perform the copy benchmark.
    for (auto _ : state) {
        host_copy(get_data(source), dest)->ignore();
    }
}
BENCHMARK(simpleAoSHostToFixedBufferCopy)->Range(1UL, 1UL << 26);

}  // namespace vecmem::benchmark
