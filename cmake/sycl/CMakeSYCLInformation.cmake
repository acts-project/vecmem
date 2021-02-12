# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up how SYCL object file compilation should go.
set( CMAKE_SYCL_COMPILE_OBJECT
   "<CMAKE_SYCL_COMPILER> -fsycl -fsycl-targets=${CMAKE_SYCL_TARGETS} -x c++ <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>" )

# Set up how shared library building should go.
set( CMAKE_SHARED_LIBRARY_SONAME_SYCL_FLAG
   "${CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG}" )
set( CMAKE_SYCL_CREATE_SHARED_LIBRARY
   "${CMAKE_SYCL_HOST_LINKER} -fsycl -fsycl-targets=${CMAKE_SYCL_TARGETS} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>" )

# Set up how module library building should go.
set( CMAKE_SYCL_CREATE_SHARED_MODULE
   "${CMAKE_SYCL_HOST_LINKER} -fsycl -fsycl-targets=${CMAKE_SYCL_TARGETS} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>" )

# Set up how executable building shoul go.
set( CMAKE_SYCL_LINK_EXECUTABLE
   "${CMAKE_SYCL_HOST_LINKER} -fsycl -fsycl-targets=${CMAKE_SYCL_TARGETS} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>" )

# Tell CMake that the information was loaded.
set( CMAKE_SYCL_INFORMATION_LOADED TRUE )
