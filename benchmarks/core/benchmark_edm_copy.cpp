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
#include "../common/simple_aos_copy_benchmarks.hpp"
#include "../common/simple_soa_copy_benchmarks.hpp"

// Google benchmark include(s).
#include <benchmark/benchmark.h>

namespace vecmem::benchmark {

/// The (host) memory resource to use in the benchmark(s).
static host_memory_resource host_mr;
/// The copy object to use in the benchmark(s).
static copy host_copy;

BENCHMARK_CAPTURE(simple_soa_direct_h2d_copy_benchmark, host_fixed_buffer,
                  host_mr, host_mr, host_copy, data::buffer_type::fixed_size)
    ->Range(1UL, 1UL << 26);
BENCHMARK_CAPTURE(simple_soa_direct_h2d_copy_benchmark, host_resizable_buffer,
                  host_mr, host_mr, host_copy, data::buffer_type::resizable)
    ->Range(1UL, 1UL << 26);

BENCHMARK_CAPTURE(simple_soa_optimal_h2d_copy_benchmark, host_fixed_buffer,
                  host_mr, host_mr, host_copy, host_copy,
                  data::buffer_type::fixed_size)
    ->Range(1UL, 1UL << 26);
BENCHMARK_CAPTURE(simple_soa_optimal_h2d_copy_benchmark, host_resizable_buffer,
                  host_mr, host_mr, host_copy, host_copy,
                  data::buffer_type::resizable)
    ->Range(1UL, 1UL << 26);

BENCHMARK_CAPTURE(simple_soa_direct_d2h_copy_benchmark, host, host_mr, host_mr,
                  host_copy)
    ->Range(1UL, 1UL << 26);
BENCHMARK_CAPTURE(simple_soa_optimal_d2h_copy_benchmark, host, host_mr, host_mr,
                  host_copy, host_copy)
    ->Range(1UL, 1UL << 26);

BENCHMARK_CAPTURE(simple_aos_h2d_copy_benchmark, host_fixed_buffer, host_mr,
                  host_mr, host_copy, data::buffer_type::fixed_size)
    ->Range(1UL, 1UL << 26);
BENCHMARK_CAPTURE(simple_aos_h2d_copy_benchmark, host_resizable_buffer, host_mr,
                  host_mr, host_copy, data::buffer_type::resizable)
    ->Range(1UL, 1UL << 26);
BENCHMARK_CAPTURE(simple_aos_d2h_copy_benchmark, host, host_mr, host_mr,
                  host_copy)
    ->Range(1UL, 1UL << 26);

}  // namespace vecmem::benchmark
