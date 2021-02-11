# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Set up the used C++ standard(s).
set( CMAKE_HIP_STANDARD 14 CACHE STRING "The (HIP) C++ standard to use" )
