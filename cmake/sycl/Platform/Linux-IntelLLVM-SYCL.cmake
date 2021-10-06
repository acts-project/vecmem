# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( Platform/Linux-IntelLLVM )

# Set up the variables specifying the command line arguments of the compiler.
__linux_compiler_intel_llvm( SYCL )
