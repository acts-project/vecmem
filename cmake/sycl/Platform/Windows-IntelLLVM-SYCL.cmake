# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( Platform/Windows-IntelLLVM )

# Set up the variables specifying the command line arguments of the compiler.
__windows_compiler_intel( SYCL )

# Tweak the compiler command, to let the compiler explicitly know that it is
# receiving C++ source code with the provided .sycl file(s).
string( REPLACE "<SOURCE>" "/Tp <SOURCE>" CMAKE_SYCL_COMPILE_OBJECT
   "${CMAKE_SYCL_COMPILE_OBJECT}" )

# Tweak the linker commands to use the DPC++ executable for linking, and to
# pass the arguments to the linker correctly.
foreach( linker_command "CMAKE_SYCL_CREATE_SHARED_LIBRARY"
   "CMAKE_SYCL_CREATE_SHARED_MODULE" "CMAKE_SYCL_LINK_EXECUTABLE" )

   # Replace the VS linker with DPC++.
   string( REPLACE "<CMAKE_LINKER>" "\"${CMAKE_SYCL_HOST_LINKER}\""
      ${linker_command} "${${linker_command}}" )

   # Prefix the linker-specific arguments with "/link", to let DPC++ know
   # that these are to be given to the linker. "/out" just happens to be the
   # first linker argument on the command line. (With CMake 3.21.) So this part
   # may need to be tweaked in the future.
   string( REPLACE "/out" "/link /out"
      ${linker_command} "${${linker_command}}" )

endforeach()

# Set the flags controlling the C++ standard used by the SYCL compiler.
set( CMAKE_SYCL17_STANDARD_COMPILE_OPTION "/std:c++17" )
set( CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "/std:c++17" )
