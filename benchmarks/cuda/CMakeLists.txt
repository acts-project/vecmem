# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-cuda )

# Set up the benchmark(s) for the CUDA library.
add_executable( vecmem_benchmark_cuda
    "benchmark_cuda.cpp"
    "benchmark_copy.cpp" )

target_link_libraries(
    vecmem_benchmark_cuda

    PRIVATE
    vecmem::cuda
    vecmem_benchmark_common
    benchmark::benchmark
    benchmark::benchmark_main
)
