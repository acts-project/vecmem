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
