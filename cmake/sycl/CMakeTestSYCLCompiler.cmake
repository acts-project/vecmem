# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Start with the correct status message.
PrintTestCompilerStatus( "SYCL" )

# Try to use the HIP compiler.
file( WRITE
   "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/main.sycl"
   "#include <CL/sycl.hpp>\n"
   "int main() { return 0; }\n" )
try_compile( CMAKE_SYCL_COMPILER_WORKS "${CMAKE_BINARY_DIR}"
   "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/main.sycl"
   OUTPUT_VARIABLE __CMAKE_SYCL_COMPILER_OUTPUT )

# Move the results of the test into a regular variable.
set( CMAKE_SYCL_COMPILER_WORKS ${CMAKE_SYCL_COMPILER_WORKS} )
unset( CMAKE_SYCL_COMPILER_WORKS CACHE )

# Check the results of the test.
if( NOT CMAKE_SYCL_COMPILER_WORKS )
   PrintTestCompilerResult( CHECK_FAIL "broken" )
   message( FATAL_ERROR "The SYCL compiler\n"
      "  \"${CMAKE_SYCL_COMPILER}\"\n"
      "is not able to compile a simple test program.\n"
      "It fails with the following output:\n"
      "  ${__CMAKE_SYCL_COMPILER_OUTPUT}\n\n"
      "CMake will not be able to correctly generate this project." )
endif()
PrintTestCompilerResult( CHECK_PASS "works" )
