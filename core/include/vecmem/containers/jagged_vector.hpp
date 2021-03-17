/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/containers/vector.hpp"

namespace vecmem {
    template<typename T>
    using jagged_vector = vector<vector<T>>;
}
