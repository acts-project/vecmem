/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/sycl/async_copy.hpp>
#include <vecmem/utils/sycl/copy.hpp>

// Common benchmark include(s).
#include "../common/simple_aos_copy_benchmarks.hpp"
#include "../common/simple_soa_copy_benchmarks.hpp"

// Google benchmark include(s).
#include <benchmark/benchmark.h>

namespace vecmem::benchmark {

/// SYCL queue to use in the benchmark(s).
static sycl::queue_wrapper queue;

/// Non-pinned host memory resource to use in the benchmark(s).
static host_memory_resource host_mr;
/// Pinned host memory resource to use in the benchmark(s).
static sycl::host_memory_resource sycl_host_mr{queue};

/// Device memory resource to use in the benchmark(s).
static sycl::device_memory_resource device_mr{queue};

/// The host copy object to use in the benchmark(s).
static copy host_copy;
/// The synchronous device copy object to use in the benchmark(s).
static sycl::copy device_copy{queue};
/// The asynchronous device copy object to use in the benchmark(s).
static sycl::async_copy async_device_copy{queue};

//
// Helper macro(s) for setting up all the different benchmarks.
//
#define CONFIGURE_BENCHMARK(BM) BM->Range(1UL, 1UL << 26)

#define EDM_COPY_BENCHMARKS(TITLE, HOST_MR, DEVICE_MR, HOST_COPY, DEVICE_COPY) \
    CONFIGURE_BENCHMARK(BENCHMARK_CAPTURE(                                     \
        simple_soa_direct_h2d_copy_benchmark, TITLE##_fixed_buffer, HOST_MR,   \
        DEVICE_MR, DEVICE_COPY, data::buffer_type::fixed_size));               \
    CONFIGURE_BENCHMARK(BENCHMARK_CAPTURE(                                     \
        simple_soa_direct_h2d_copy_benchmark, TITLE##_resizable_buffer,        \
        HOST_MR, DEVICE_MR, DEVICE_COPY, data::buffer_type::resizable));       \
    CONFIGURE_BENCHMARK(BENCHMARK_CAPTURE(                                     \
        simple_soa_optimal_h2d_copy_benchmark, TITLE##_fixed_buffer, HOST_MR,  \
        DEVICE_MR, HOST_COPY, DEVICE_COPY, data::buffer_type::fixed_size));    \
    CONFIGURE_BENCHMARK(BENCHMARK_CAPTURE(                                     \
        simple_soa_optimal_h2d_copy_benchmark, TITLE##_resizable_buffer,       \
        HOST_MR, DEVICE_MR, HOST_COPY, DEVICE_COPY,                            \
        data::buffer_type::resizable));                                        \
    CONFIGURE_BENCHMARK(                                                       \
        BENCHMARK_CAPTURE(simple_soa_direct_d2h_copy_benchmark, TITLE,         \
                          HOST_MR, DEVICE_MR, DEVICE_COPY));                   \
    CONFIGURE_BENCHMARK(                                                       \
        BENCHMARK_CAPTURE(simple_soa_optimal_d2h_copy_benchmark, TITLE,        \
                          HOST_MR, DEVICE_MR, HOST_COPY, DEVICE_COPY));        \
    CONFIGURE_BENCHMARK(BENCHMARK_CAPTURE(                                     \
        simple_aos_h2d_copy_benchmark, TITLE##_fixed_buffer, HOST_MR,          \
        DEVICE_MR, DEVICE_COPY, data::buffer_type::fixed_size));               \
    CONFIGURE_BENCHMARK(BENCHMARK_CAPTURE(                                     \
        simple_aos_h2d_copy_benchmark, TITLE##_resizable_buffer, HOST_MR,      \
        DEVICE_MR, DEVICE_COPY, data::buffer_type::resizable));                \
    CONFIGURE_BENCHMARK(BENCHMARK_CAPTURE(simple_aos_d2h_copy_benchmark,       \
                                          TITLE, HOST_MR, DEVICE_MR,           \
                                          DEVICE_COPY))

//
// Set up all the different benchmarks.
//
EDM_COPY_BENCHMARKS(sycl_pageable_sync, host_mr, device_mr, host_copy,
                    device_copy);
EDM_COPY_BENCHMARKS(sycl_pageable_async, host_mr, device_mr, host_copy,
                    async_device_copy);
EDM_COPY_BENCHMARKS(sycl_pinned_sync, sycl_host_mr, device_mr, host_copy,
                    device_copy);
EDM_COPY_BENCHMARKS(sycl_pinned_async, sycl_host_mr, device_mr, host_copy,
                    async_device_copy);

}  // namespace vecmem::benchmark
