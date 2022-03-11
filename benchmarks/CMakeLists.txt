# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up Google Benchmark.
option(
   VECMEM_SETUP_GOOGLEBENCHMARK
   "Set up the Google Benchmark targets explicitly"
   TRUE
)
option(
   VECMEM_USE_SYSTEM_GOOGLEBENCHMARK
   "Pick up an existing installation of Google Benchmark from the build environment"
   FALSE
)
if(VECMEM_SETUP_GOOGLEBENCHMARK)
   if(VECMEM_USE_SYSTEM_GOOGLEBENCHMARK)
      find_package(benchmark REQUIRED)
   else()
      add_subdirectory(googlebenchmark)
   endif()
endif()

# Include the library specific tests.
add_subdirectory(core)
if(VECMEM_BUILD_CUDA_LIBRARY)
   add_subdirectory(cuda)
endif()
