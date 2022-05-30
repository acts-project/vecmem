/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// Common benchmark include(s).
#include "../common/make_jagged_vector.hpp"

// Google benchmark include(s).
#include <benchmark/benchmark.h>

// System include(s).
#include <vector>

namespace vecmem::cuda::benchmark {

/// The (managed) memory resource to use in the benchmark(s).
static managed_memory_resource managed_mr;
/// The copy object to use in the benchmark(s).
static copy cuda_copy;

/// Function benchmarking the @c vecmem::cuda::copy jagged vector operations
void jaggedVectorUnknownCopy(::benchmark::State& state) {

    // Create the "source vector".
    jagged_vector<int> source = vecmem::benchmark::make_jagged_vector(
        state.range(0), state.range(1), managed_mr);
    const data::jagged_vector_data<int> source_data = get_data(source);
    // Create the "destination vector".
    jagged_vector<int> dest;

    // Perform the copy benchmark.
    for (auto _ : state) {
        dest.clear();
        cuda_copy(source_data, dest);
    }
}
// Set up the benchmark.
BENCHMARK(jaggedVectorUnknownCopy)->Ranges({{10, 100000}, {50, 5000}});

}  // namespace vecmem::cuda::benchmark
