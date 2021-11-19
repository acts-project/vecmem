add_executable(vecmem_benchmark_core "benchmark_core.cpp")

target_link_libraries(
    vecmem_benchmark_core
    
    PRIVATE
    vecmem::core
    benchmark::benchmark
    benchmark::benchmark_main
)
