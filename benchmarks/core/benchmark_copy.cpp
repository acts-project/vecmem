/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// Common benchmark include(s).
#include "../common/make_jagged_vector.hpp"

// Google benchmark include(s).
#include <benchmark/benchmark.h>

// System include(s).
#include <vector>

namespace vecmem::benchmark {

/// The (host) memory resource to use in the benchmark(s).
static host_memory_resource host_mr;
/// The copy object to use in the benchmark(s).
static copy host_copy;

/// Function benchmarking the @c vecmem::copy jagged vector operations
void jaggedVectorHostCopy(::benchmark::State& state) {

    // Create the "source vector".
    jagged_vector<int> source =
        make_jagged_vector(state.range(0), state.range(1), host_mr);
    const data::jagged_vector_data<int> source_data = get_data(source);
    // Create the "destination vector".
    jagged_vector<int> dest;

    // Perform the copy benchmark.
    for (auto _ : state) {
        dest.clear();
        host_copy(source_data, dest);
    }
}
// Set up the benchmark.
BENCHMARK(jaggedVectorHostCopy)->Ranges({{10, 100000}, {50, 5000}});

}  // namespace vecmem::benchmark
