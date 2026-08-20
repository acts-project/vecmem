/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "simple_soa_copy_benchmarks.hpp"

#include "setup_simple_copy_counters.hpp"
#include "simple_soa_container.hpp"

namespace vecmem::benchmark {

void simple_soa_direct_h2d_copy_benchmark(::benchmark::State& state,
                                          memory_resource& host_mr,
                                          memory_resource& device_mr,
                                          copy& device_copy,
                                          data::buffer_type buffer_type) {

    // Get the size of the host container to create, while setting up the
    // counters used by the benchmark.
    const std::size_t size = setup_simple_copy_counters(state);

    // Create the source host container.
    simple_soa_container::host source{host_mr};
    source.resize(size);

    // Create the destination (host) buffer.
    simple_soa_container::buffer dest{
        static_cast<simple_soa_container::buffer::size_type>(size), device_mr,
        buffer_type};
    device_copy.setup(dest)->wait();

    // Perform the copy benchmark.
    for (auto _ : state) {
        device_copy(get_data(source), dest)->wait();
    }
}

void simple_soa_optimal_h2d_copy_benchmark(::benchmark::State& state,
                                           memory_resource& host_mr,
                                           memory_resource& device_mr,
                                           copy& host_copy, copy& device_copy,
                                           data::buffer_type buffer_type) {

    // Get the size of the host container to create, while setting up the
    // counters used by the benchmark.
    const std::size_t size = setup_simple_copy_counters(state);

    // Create the source host container and buffer.
    simple_soa_container::host host{host_mr};
    host.resize(size);
    simple_soa_container::buffer source{
        static_cast<simple_soa_container::buffer::size_type>(size), host_mr,
        buffer_type};
    host_copy.setup(source)->wait();
    host_copy(get_data(host), source)->wait();

    // Create the destination (host) buffer.
    simple_soa_container::buffer dest{
        static_cast<simple_soa_container::buffer::size_type>(size), device_mr,
        buffer_type};
    device_copy.setup(dest)->wait();

    // Perform the copy benchmark.
    for (auto _ : state) {
        device_copy(source, dest)->wait();
    }
}

void simple_soa_direct_d2h_copy_benchmark(::benchmark::State& state,
                                          memory_resource& host_mr,
                                          memory_resource& device_mr,
                                          copy& device_copy) {

    // Get the size of the host container to create, while setting up the
    // counters used by the benchmark.
    const std::size_t size = setup_simple_copy_counters(state);

    // Create the source buffer.
    simple_soa_container::buffer source{
        static_cast<simple_soa_container::buffer::size_type>(size), device_mr};
    device_copy.setup(source)->wait();

    // Create the destination (host) container.
    simple_soa_container::host dest{host_mr};

    // Perform the copy benchmark.
    for (auto _ : state) {
        state.PauseTiming();
        dest.resize(0u);
        state.ResumeTiming();
        device_copy(source, dest)->wait();
    }
}

void simple_soa_optimal_d2h_copy_benchmark(::benchmark::State& state,
                                           memory_resource& host_mr,
                                           memory_resource& device_mr,
                                           copy& host_copy, copy& device_copy) {

    // Get the size of the host container to create, while setting up the
    // counters used by the benchmark.
    const std::size_t size = setup_simple_copy_counters(state);

    // Create the source buffer.
    simple_soa_container::buffer source{
        static_cast<simple_soa_container::buffer::size_type>(size), device_mr};
    device_copy.setup(source)->wait();

    // Create the destination buffer.
    simple_soa_container::buffer dest{
        static_cast<simple_soa_container::buffer::size_type>(size), host_mr};
    host_copy.setup(dest)->wait();

    // Perform the copy benchmark.
    for (auto _ : state) {
        device_copy(source, dest)->wait();
    }
}

}  // namespace vecmem::benchmark
