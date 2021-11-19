add_executable(vecmem_benchmark_cuda "benchmark_cuda.cpp")

target_link_libraries(
    vecmem_benchmark_cuda
    
    PRIVATE
    vecmem::core
    vecmem::cuda
    benchmark::benchmark
    benchmark::benchmark_main
)
