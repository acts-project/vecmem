# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include whatever CMake can give us to configure the LLVM based intel compiler,
# and use it.
include( Platform/Linux-IntelLLVM OPTIONAL
   RESULT_VARIABLE IntelLLVM_AVAILABLE )
if( IntelLLVM_AVAILABLE )
   # We have a "new enough" version of CMake to use the most appropriate
   # configuration.
   __linux_compiler_intel_llvm( SYCL )
else()
   # We have a somewhat older version of CMake. Use the configuration for the
   # older Intel compilers.
   include( Platform/Linux-Intel OPTIONAL
      RESULT_VARIABLE Intel_AVAILABLE )
   if( Intel_AVAILABLE )
      __linux_compiler_intel( SYCL )
   endif()
endif()

# Tweak the compiler command, to let the compiler explicitly know that it is
# receiving C++ source code with the provided .sycl file(s).
string( REPLACE "<SOURCE>" "-x c++ <SOURCE>" CMAKE_SYCL_COMPILE_OBJECT
   "${CMAKE_SYCL_COMPILE_OBJECT}" )

# Set the flags controlling the C++ standard used by the SYCL compiler.
set( CMAKE_SYCL17_STANDARD_COMPILE_OPTION "-std=c++17" )
set( CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "-std=c++17" )
