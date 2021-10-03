# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( Platform/Linux-IntelLLVM )

# Set up the variables specifying the command line arguments of the compiler.
__linux_compiler_intel_llvm( SYCL )

# Tweak the compiler command, to let the compiler explicitly know that it is
# receiving C++ source code with the provided .sycl file(s).
string( REPLACE "<SOURCE>" "-x c++ <SOURCE>" CMAKE_SYCL_COMPILE_OBJECT
   "${CMAKE_SYCL_COMPILE_OBJECT}" )

# Set the flags controlling the C++ standard used by the SYCL compiler.
set( CMAKE_SYCL17_STANDARD_COMPILE_OPTION "-std=c++17" )
set( CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "-std=c++17" )
