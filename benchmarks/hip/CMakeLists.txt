# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-hip )

# Set up the benchmark(s) for the HIP library.
add_executable( vecmem_benchmark_hip
    "benchmark_hip.cpp"
    "benchmark_copy.cpp"
    "benchmark_edm_copy.cpp" )

target_link_libraries(
    vecmem_benchmark_hip

    PRIVATE
    vecmem::core
    vecmem::hip
    vecmem_benchmark_common
    benchmark::benchmark
    benchmark::benchmark_main
)

set_target_properties( vecmem_benchmark_hip PROPERTIES
   FOLDER "vecmem/benchmarks" )
