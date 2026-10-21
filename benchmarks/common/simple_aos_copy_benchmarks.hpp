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

/// Benchmark copying @c simple_aos_container::host to
/// @c simple_aos_container::buffer
void simple_aos_h2d_copy_benchmark(::benchmark::State& state,
                                   memory_resource& host_mr,
                                   memory_resource& device_mr,
                                   copy& device_copy,
                                   data::buffer_type buffer_type);

/// Benchmark copying @c simple_aos_container::buffer to
/// @c simple_aos_container::host
void simple_aos_d2h_copy_benchmark(::benchmark::State& state,
                                   memory_resource& host_mr,
                                   memory_resource& device_mr,
                                   copy& device_copy);

}  // namespace vecmem::benchmark
