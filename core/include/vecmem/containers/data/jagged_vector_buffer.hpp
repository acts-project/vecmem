/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/memory/deallocator.hpp"
#include "vecmem/memory/memory_resource.hpp"

// System include(s).
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace vecmem::data {

   /// Object owning all the data of a jagged vector
   ///
   /// This type is needed for the explicit memory management of jagged vectors.
   ///
   template< typename TYPE >
   class jagged_vector_buffer : public jagged_vector_view< TYPE > {

   public:
      /// The base type used by this class
      typedef jagged_vector_view< TYPE > base_type;
      /// Use the base class's @c size_type
      typedef typename base_type::size_type size_type;
      /// Use the base class's @c value_type
      typedef typename base_type::value_type value_type;
      /// Pointer type to the jagged array
      typedef typename base_type::pointer pointer;

      /// @name Checks on the type of the array element
      /// @{

      /// Make sure that the template type does not have a custom destructor
      static_assert( std::is_trivially_destructible< TYPE >::value,
                     "vecmem::data::jagged_vector_buffer can not handle types "
                     "with custom destructors" );
      /// Make sure that @c vecmem::data::vector_view does not have a custom
      /// destructor
      static_assert( std::is_trivially_destructible< value_type >::value,
                     "vecmem::data::jagged_vector_buffer can not handle types "
                     "with custom destructors" );

      /// @}

      /// Constructor from an existing @c vecmem::data::jagged_vector_view
      ///
      /// @param other The existing @c vecmem::data::jagged_vector_view object
      ///        that this buffer should mirror.
      /// @param resource The device accessible memory resource, which may also
      ///        be host accessible.
      /// @param host_access_resource An optional host accessible memory
      ///        resource. Needed if @c resource is not host accessible.
      template< typename OTHERTYPE >
      jagged_vector_buffer( const jagged_vector_view< OTHERTYPE >& other,
                            memory_resource& resource,
                            memory_resource* host_access_resource = nullptr );

      /// Constructor from a vector of ("inner vector") sizes
      ///
      /// @param sizes Simple vector holding the sizes of the "inner vectors"
      ///        for the jagged vector buffer.
      /// @param resource The device accessible memory resource, which may also
      ///        be host accessible.
      /// @param host_access_resource An optional host accessible memory
      ///        resource. Needed if @c resource is not host accessible.
      jagged_vector_buffer( const std::vector< std::size_t >& sizes,
                            memory_resource& resource,
                            memory_resource* host_access_resource = nullptr );

      /// Access the host accessible array describing the inner vectors
      ///
      /// This may or may not return the same pointer that
      /// @c vecmem::data::jagged_vector_view::m_ptr holds. If the buffer is set
      /// up on top of a "shared" (both host- and device accessible) memory
      /// resource, then the two will be the same. If not, then
      /// @c vecmem::data::jagged_vector_view::m_ptr is set up to point at the
      /// device accessible array, and this function returns a pointer to the
      /// host accessible one.
      ///
      pointer host_ptr() const;

   private:
      /// Data object for the @c vecmem::data::vector_view array
      std::unique_ptr< value_type, details::deallocator > m_outer_memory;
      /// Data object for the @c vecmem::data::vector_view array on the host
      std::unique_ptr< value_type, details::deallocator > m_outer_host_memory;
      /// Data object owning the memory of the "inner vectors"
      std::unique_ptr< TYPE, details::deallocator > m_inner_memory;

   }; // class jagged_vector_buffer

} // namespace vecmem::data

// Include the implementation.
#include "vecmem/containers/impl/jagged_vector_buffer.ipp"
