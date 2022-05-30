/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>

namespace vecmem::benchmark {

/// Function creating a jagged vector with some general size specifications
///
/// It creates a jagged vector with a fixed "outer size", and random sized
/// "inner vectors" that would not be larger than some specified value.
///
/// @param outerSize The fixed "outer size" of the resulting vector
/// @param maxInnerSize The maximum for the random "inner sizes" of the
///                     resulting vector
/// @param mr The memory resource to use
/// @return A jagged vector with the specifier properties
///
jagged_vector<int> make_jagged_vector(std::size_t outerSize,
                                      std::size_t maxInnerSize,
                                      memory_resource& mr);

}  // namespace vecmem::benchmark
