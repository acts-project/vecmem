# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# CMake include(s).
include( CMakeParseArguments )
include( GenerateExportHeader )

# Helper function for setting up the VecMem libraries.
#
# Usage: vecmem_add_library( vecmem_core core
#                            [TYPE SHARED/STATIC]
#                            include/source1.hpp source2.cpp )
#
function( vecmem_add_library fullname basename )

   # Parse the function's options.
   cmake_parse_arguments( ARG "" "TYPE" "" ${ARGN} )

   # Create the library.
   add_library( ${fullname} ${ARG_TYPE} ${ARG_UNPARSED_ARGUMENTS} )

   # Set up symbol exports for the library.
   set( export_header_filename
      "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecmem/${fullname}_export.hpp" )
   generate_export_header( ${fullname}
      EXPORT_FILE_NAME "${export_header_filename}" )
   install( FILES "${export_header_filename}"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/vecmem" )
   target_include_directories( ${fullname} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}> )

   # Set up how clients should find its headers.
   target_include_directories( ${fullname} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

   # Make sure that the library is available as "vecmem::${basename}" in every
   # situation.
   set_target_properties( ${fullname} PROPERTIES EXPORT_NAME ${basename} )
   add_library( vecmem::${basename} ALIAS ${fullname} )

   # Specify the (SO)VERSION of the library.
   set_target_properties( ${fullname} PROPERTIES
      VERSION ${PROJECT_VERSION}
      SOVERSION ${PROJECT_VERSION_MAJOR} )

   # Set up the installation of the library and its headers.
   install( TARGETS ${fullname}
      EXPORT vecmem-exports
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" )
   install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )

endfunction( vecmem_add_library )

# Helper function for setting up the VecMem tests.
#
# Usage: vecmem_add_test( core_containers source1.cpp source2.cpp
#                         LINK_LIBRARIES vecmem::core )
#
function( vecmem_add_test name )

   # Parse the function's options.
   cmake_parse_arguments( ARG "" "" "LINK_LIBRARIES" ${ARGN} )

   # Create the test executable.
   set( test_exe_name "vecmem_test_${name}" )
   add_executable( ${test_exe_name} ${ARG_UNPARSED_ARGUMENTS} )
   if( ARG_LINK_LIBRARIES )
      target_link_libraries( ${test_exe_name} PRIVATE ${ARG_LINK_LIBRARIES} )
   endif()

   # Run the executable as the test.
   add_test( NAME ${test_exe_name}
      COMMAND $<TARGET_FILE:${test_exe_name}> )

endfunction( vecmem_add_test )

# Helper function for adding individual flags to "flag variables".
#
# Usage: vecmem_add_flag( CMAKE_CXX_FLAGS "-Wall" )
#
function( vecmem_add_flag name value )

   # Escape special characters in the value:
   set( matchedValue "${value}" )
   foreach( c "*" "." "^" "$" "+" "?" )
      string( REPLACE "${c}" "\\${c}" matchedValue "${matchedValue}" )
   endforeach()

   # Check if the variable already has this value in it:
   if( "${${name}}" MATCHES "${matchedValue}" )
      return()
   endif()

   # If not, then let's add it now:
   set( ${name} "${${name}} ${value}" PARENT_SCOPE )

endfunction( vecmem_add_flag )
