# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( Platform/Windows-IntelLLVM )

# Set up the variables specifying the command line arguments of the compiler.
__windows_compiler_intel( SYCL )
