/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/deallocator.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"

// System include(s).
#include <memory>

namespace vecmem::data {

    /**
     * @brief A data wrapper for jagged vectors.
     *
     * This class constructs the relevant administrative data from a vector of
     * vectors, and is designed to be later turned into a @c jagged_vector_view
     * object.
     */
    template<typename T>
    class jagged_vector_data : public jagged_vector_view<T> {

    public:
        /// Type of the base class
        using base_type = jagged_vector_view<T>;
        /// Use the base class's @c size_type
        typedef typename base_type::size_type size_type;
        /// Use the base class's @c value_type
        typedef typename base_type::value_type value_type;

        /**
         * @brief Construct jagged vector data from raw information
         *
         * This class converts from std vectors (or rather, vecmem::vectors) to
         * a jagged vector data.
         *
         * @param[in] size Size of the "outer vector"
         * @param[in] mem The memory resource to manage the internal state
         */
        jagged_vector_data(
            size_type size,
            memory_resource& mem
        );

    private:
        /// Data object owning the allocated memory
        std::unique_ptr< value_type, details::deallocator > m_memory;

    }; // class jagged_vector_data

} // namespace vecmem::data

// Include the implementation.
#include "vecmem/containers/impl/jagged_vector_data.ipp"
