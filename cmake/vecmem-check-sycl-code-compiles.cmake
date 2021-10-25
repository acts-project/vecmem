# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Helper function for checking if some SYCL files can be built into an
# executable.
function( vecmem_check_sycl_code_compiles _variable )

   # Enable the SYCL language.
   enable_language( SYCL )

   # Return early, if the result variable already has a value.
   if( DEFINED ${_variable} )
      return()
   endif()

   # Print a greeting message.
   if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
      message( STATUS "Performing Test ${_variable}" )
   else()
      message( CHECK_START "Performing Test ${_variable}" )
   endif()

   # Perform the build attempt.
   try_compile( ${_variable}
      "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${_variable}"
      SOURCES ${ARGN} )

   # Print a result message.
   if( ${_variable} )
      if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
         message( STATUS "Performing Test ${_variable} - Success" )
      else()
         message( CHECK_PASS "Success" )
      endif()
   else()
      if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
         message( STATUS "Performing Test ${_variable} - Failed" )
      else()
         message( CHECK_FAIL "Failed" )
      endif()
   endif()

endfunction( vecmem_check_sycl_code_compiles )
