# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the benchmark(s) for the core library.
add_executable(vecmem_benchmark_core "benchmark_core.cpp")

target_link_libraries(
    vecmem_benchmark_core

    PRIVATE
    vecmem::core
    benchmark::benchmark
    benchmark::benchmark_main
)
