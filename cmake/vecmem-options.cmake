# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Look for the supported GPU languages. But only if the user didn't specify
# explicitly if (s)he wants them used. If they are turned on/off through
# cache variables explicitly, then skip looking for them at this point.
include( vecmem-check-language )
foreach( lang CUDA HIP SYCL )
   if( NOT DEFINED VECMEM_BUILD_${lang}_LIBRARY )
      vecmem_check_language( ${lang} )
   endif()
endforeach()

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

# Set the default library type to build.
set( BUILD_SHARED_LIBS TRUE CACHE BOOL
   "Flag for building shared/static libraries" )
