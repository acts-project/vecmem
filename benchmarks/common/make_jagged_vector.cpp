/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "make_jagged_vector.hpp"

// System include(s).
#include <random>

namespace vecmem::benchmark {

jagged_vector<int> make_jagged_vector(std::size_t outerSize,
                                      std::size_t maxInnerSize,
                                      memory_resource& mr) {

    // Create the result object.
    jagged_vector<int> result(&mr);
    result.reserve(outerSize);

    // Set up a simple random number generator for the inner vector sizes.
    std::default_random_engine eng;
    std::uniform_int_distribution<std::size_t> gen(0, maxInnerSize);

    // Set up each of its inner vectors.
    for (std::size_t i = 0; i < outerSize; ++i) {
        result.push_back(jagged_vector<int>::value_type(gen(eng), &mr));
    }

    // Return the vector.
    return result;
}

}  // namespace vecmem::benchmark
