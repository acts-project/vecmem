# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Decide whether AMD or NVidia code is generated using HIP.
set( CMAKE_HIP_PLATFORM_DEFAULT "hcc" )
if( NOT "$ENV{HIP_PLATFORM}" STREQUAL "" )
   set( CMAKE_HIP_PLATFORM_DEFAULT "$ENV{HIP_PLATFORM}" )
endif()
set( CMAKE_HIP_PLATFORM "${CMAKE_HIP_PLATFORM_DEFAULT}" CACHE STRING
   "Platform to build the HIP code for" )
set_property( CACHE CMAKE_HIP_PLATFORM
   PROPERTY STRINGS "hcc" "nvcc" )

# Set a helper variable.
set( _quietFlag )
if( HIPToolkit_FIND_QUIETLY )
   set( _quietFlag QUIET )
endif()

# Look for the CUDA toolkit if we are building NVidia code.
set( _requiredVars )
if( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" )
   find_package( CUDAToolkit ${_quietFlag} )
   list( APPEND _requiredVars CUDAToolkit_FOUND )
endif()

# Look for the ROCm/HIP header(s).
find_path( HIP_INCLUDE_DIR
   NAMES "hip/hip_runtime.h"
         "hip/hip_runtime_api.h"
   PATHS "${HIP_ROOT_DIR}"
         ENV ROCM_PATH
         ENV HIP_PATH
         "/opt/rocm"
         "/opt/rocm/hip"
   PATH_SUFFIXES "include"
   DOC "ROCm/HIP include directory" )
mark_as_advanced( HIP_INCLUDE_DIR )
set( HIP_INCLUDE_DIRS "${HIP_INCLUDE_DIR}" )
list( APPEND _requiredVars HIP_INCLUDE_DIR )

# Figure out the version of HIP.
find_file( HIP_VERSION_FILE
   NAMES "hip/hip_version.h"
   HINTS "${HIP_INCLUDE_DIR}"
   DOC "Path to hip/hip_version.h" )
mark_as_advanced( HIP_VERSION_FILE )
if( HIP_VERSION_FILE )
   file( READ "${HIP_VERSION_FILE}" _versionFileContents )
   if( "${_versionFileContents}" MATCHES "HIP_VERSION_MAJOR ([0-9]+)" )
      set( HIP_VERSION_MAJOR "${CMAKE_MATCH_1}" )
   endif()
   if( "${_versionFileContents}" MATCHES "HIP_VERSION_MINOR ([0-9]+)" )
      set( HIP_VERSION_MINOR "${CMAKE_MATCH_1}" )
   endif()
   if( "${_versionFileContents}" MATCHES "HIP_VERSION_PATCH ([0-9]+)" )
      set( HIP_VERSION_PATCH "${CMAKE_MATCH_1}" )
   endif()
   unset( _versionFileContents )
   set( HIP_VERSION
        "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH}" )
endif()

# Look for the HIP runtime library.
set( HIP_LIBRARIES )
if( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" )
   find_library( HIP_amdhip64_LIBRARY
      NAMES "amdhip64"
      PATHS "${HIP_ROOT_DIR}"
            ENV ROCM_PATH
            ENV HIP_PATH
            "/opt/rocm"
            "/opt/rocm/hip"
      PATH_SUFFIXES "lib" "lib64"
      DOC "AMD/HIP Runtime Library" )
   mark_as_advanced( HIP_amdhip64_LIBRARY )
   set( HIP_RUNTIME_LIBRARY "${HIP_amdhip64_LIBRARY}" )
   list( APPEND HIP_LIBRARIES "${HIP_amdhip64_LIBRARY}" )
   list( APPEND _requiredVars HIP_RUNTIME_LIBRARY )
elseif( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" )
   set( HIP_RUNTIME_LIBRARY "${CUDA_cudart_LIBRARY}" )
   list( APPEND HIP_LIBRARIES CUDA::cudart )
else()
   message( SEND_ERROR "Invalid (CMAKE_)HIP_PLATFORM setting received" )
endif()

# Set up the compiler definitions needed to use the HIP headers.
if( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" )
   set( HIP_DEFINITIONS "__HIP_PLATFORM_HCC__" )
elseif( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" )
   set( HIP_DEFINITIONS "__HIP_PLATFORM_NVCC__" )
else()
   message( SEND_ERROR "Invalid (CMAKE_)HIP_PLATFORM setting received" )
endif()

# Handle the standard find_package arguments:
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( HIPToolkit
   FOUND_VAR HIPToolkit_FOUND
   REQUIRED_VARS HIP_INCLUDE_DIR ${_requiredVars}
   VERSION_VAR HIP_VERSION )

# Set up the imported target(s).
if( NOT TARGET HIP::hiprt )
   add_library( HIP::hiprt UNKNOWN IMPORTED )
   set_target_properties( HIP::hiprt PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${HIP_RUNTIME_LIBRARY}"
      INTERFACE_LINK_LIBRARIES "${HIP_LIBRARIES}"
      INTERFACE_COMPILE_DEFINITIONS "${HIP_DEFINITIONS}" )
endif()

# Clean up.
unset( _quietFlag )
unset( _requiredVars )
