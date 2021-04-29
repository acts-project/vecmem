# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Look for the supported GPU languages.
include( vecmem-check-language )
vecmem_check_language( CUDA )
vecmem_check_language( HIP )
vecmem_check_language( SYCL )

# Helper function for setting up the library building flags.
function( vecmem_lib_option language descr )

   # Figure out what the default value should be for the variable.
   set( _default FALSE )
   if( CMAKE_${language}_COMPILER )
      set( _default TRUE )
   endif()

   # Set up the configuration option.
   option( VECMEM_BUILD_${language}_LIBRARY "${descr}" ${_default} )

endfunction( vecmem_lib_option )

# Flag specifying whether CUDA support should be built.
vecmem_lib_option( CUDA "Build the vecmem::cuda library" )

# Flag specifying whether HIP support should be built.
vecmem_lib_option( HIP "Build the vecmem::hip library" )

# Flag specifying whether SYCL support should be built.
vecmem_lib_option( SYCL "Build the vecmem::sycl library" )

# Debug message output level in the code.
set( VECMEM_DEBUG_MSG_LVL 0 CACHE STRING
   "Debug message output level" )
