# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( Platform/Windows-IntelLLVM )

# Set up the variables specifying the command line arguments of the compiler.
__windows_compiler_intel( SYCL )

# Tweak the MSVC linker flags, to make them compatible with DPC++.
if( ${CMAKE_VERSION} VERSION_LESS 3.21 )
   set( CMAKE_CREATE_WIN32_EXE "-Xlinker /subsystem:windows" )
   set( CMAKE_CREATE_CONSOLE_EXE "-Xlinker /subsystem:console" )
else()
   set( CMAKE_SYCL_CREATE_WIN32_EXE "-Xlinker /subsystem:windows" )
   set( CMAKE_SYCL_CREATE_CONSOLE_EXE "-Xlinker /subsystem:console" )
endif()
foreach( _type EXE SHARED MODULE )
   string( REGEX REPLACE "(/machine:[a-zA-Z0-9]+)" "-Xlinker \\1"
      CMAKE_${_type}_LINKER_FLAGS "${CMAKE_${_type}_LINKER_FLAGS}" )
endforeach()
