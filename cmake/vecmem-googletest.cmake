# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Look for GoogleTest.
find_package( GTest )

# If it was found, then we're finished.
if( GTest_FOUND )
   return()
endif()

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building GoogleTest as part of the project" )

# Declare where to get GoogleTest from.
FetchContent_Declare( GoogleTest
   URL "https://github.com/google/googletest/archive/release-1.11.0.tar.gz"
   URL_MD5 "e8a8df240b6938bb6384155d4c37d937" )

# Options used in the build of GoogleTest.
set( BUILD_GMOCK FALSE CACHE BOOL "Turn off the build of GMock" )
set( INSTALL_GTEST FALSE CACHE BOOL "Turn off the installation of GoogleTest" )
if( WIN32 )
   set( gtest_force_shared_crt TRUE CACHE BOOL
      "Use shared (DLL) run-time library, even with static libraries" )
endif()

# Silence some warnings with modern versions of CMake on macOS.
set( CMAKE_MACOSX_RPATH TRUE )

# Get it into the current directory.
FetchContent_Populate( GoogleTest )
add_subdirectory( "${googletest_SOURCE_DIR}" "${googletest_BINARY_DIR}"
   EXCLUDE_FROM_ALL )

# Set up aliases for the GTest targets with the same name that they have
# when we find GTest pre-installed.
add_library( GTest::gtest ALIAS gtest )
add_library( GTest::gtest_main ALIAS gtest_main )
