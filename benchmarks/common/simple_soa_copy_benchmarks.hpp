/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s).
#include "vecmem/containers/data/buffer_type.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// Google include(s).
#include <benchmark/benchmark.h>

namespace vecmem::benchmark {

/// Benchmark copying @c simple_soa_container::host to
/// @c simple_soa_container::buffer directly
void simple_soa_direct_h2d_copy_benchmark(::benchmark::State& state,
                                          memory_resource& host_mr,
                                          memory_resource& device_mr,
                                          copy& device_copy,
                                          data::buffer_type buffer_type);

/// Benchmark copying @c simple_soa_container::host to
/// @c simple_soa_container::buffer through an intermediate buffer
void simple_soa_optimal_h2d_copy_benchmark(::benchmark::State& state,
                                           memory_resource& host_mr,
                                           memory_resource& device_mr,
                                           copy& host_copy, copy& device_copy,
                                           data::buffer_type buffer_type);

/// Benchmark copying @c simple_soa_container::buffer to
/// @c simple_soa_container::host directly
void simple_soa_direct_d2h_copy_benchmark(::benchmark::State& state,
                                          memory_resource& host_mr,
                                          memory_resource& device_mr,
                                          copy& device_copy);

/// Benchmark copying @c simple_soa_container::buffer to
/// @c simple_soa_container::host through an intermediate buffer
void simple_soa_optimal_d2h_copy_benchmark(::benchmark::State& state,
                                           memory_resource& host_mr,
                                           memory_resource& device_mr,
                                           copy& host_copy, copy& device_copy);

}  // namespace vecmem::benchmark
