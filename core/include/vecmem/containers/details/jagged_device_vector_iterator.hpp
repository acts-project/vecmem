/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem { namespace details {

   /// Custom iterator for @c vecmem::jagged_device_vector
   ///
   /// In order for @c vecmem::jagged_device_vector to be able to offer
   /// iteration over its elements in an efficient and safe way, it needs to use
   /// this custom iterator type.
   ///
   /// It takes care of converting between the underlying data type and the
   /// type presented towards the users for access to the data. On top of
   /// providing all the functionality that an iterator has to.
   ///
   template< typename TYPE >
   class jagged_device_vector_iterator {

   public:
      /// @name Types describing the underlying data
      /// @{

      /// Type of the data object that we have an array of
      typedef data::vector_view< TYPE > data_type;
      /// Pointer to the data object
      typedef data_type* data_pointer;

      /// @}

      /// @name Type definitions, mimicking STL iterators
      /// @{

      /// Value type being (virtually) iterated on
      typedef device_vector< TYPE > value_type;
      /// (Pointer) Difference type
      typedef std::ptrdiff_t difference_type;
      /// Pointer type to the underlying (virtual) value
      typedef value_type* pointer;
      /// Reference type to the underlying (virtual) value
      typedef value_type& reference;

      /// @}

      /// Default constructor
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator();
      /// Constructor from an underlying data object
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator( data_pointer data );
      /// Construct from a non-const underlying data object
      template< typename OTHERTYPE,
                std::enable_if_t<
                   details::is_same_nc< TYPE, OTHERTYPE >::value,
                   bool > = true >
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator( data::vector_view< OTHERTYPE >* data );
      /// Copy constructor
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator(
         const jagged_device_vector_iterator& parent );
      /// Copy constructor
      template< typename T >
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator(
         const jagged_device_vector_iterator< T >& parent );

      /// Copy assignment operator
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator&
      operator=( const jagged_device_vector_iterator& rhs );

      /// @name Value accessor operators
      /// @{

      /// De-reference the iterator
      VECMEM_HOST_AND_DEVICE
      reference operator*() const;
      /// Use the iterator as a pointer
      VECMEM_HOST_AND_DEVICE
      pointer operator->() const;

      /// @}

      /// @name Iterator updating operators
      /// @{

      /// Decrement the underlying iterator (with '++' as a prefix)
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator& operator++();
      /// Decrement the underlying iterator (wuth '++' as a postfix)
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator operator++( int );

      /// Increment the underlying iterator (with '--' as a prefix)
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator& operator--();
      /// Increment the underlying iterator (with '--' as a postfix)
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator operator--( int );

      /// Decrement the underlying iterator by a specific value
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator operator+( difference_type n ) const;
      /// Decrement the underlying iterator by a specific value
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator& operator+=( difference_type n );

      /// Increment the underlying iterator by a specific value
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator operator-( difference_type n ) const;
      /// Increment the underlying iterator by a specific value
      VECMEM_HOST_AND_DEVICE
      jagged_device_vector_iterator& operator-=( difference_type n );

      /// @}

      /// @name Comparison operators
      /// @{

      /// Check for the equality of two iterators
      bool operator==( const jagged_device_vector_iterator& other ) const;
      /// Check for the inequality of two iterators
      bool operator!=( const jagged_device_vector_iterator& other ) const;

      /// @}

   private:
      /// Ensure that the internal value object is valid
      void ensure_valid() const;

      /// Pointer to the data (in an array)
      data_pointer m_ptr;
      /// Flag showing whether the value object is up to date with the pointer
      mutable bool m_value_is_valid;
      /// Helper object returned by the iterator
      mutable value_type m_value;

   }; // class jagged_device_vector_iterator

} } // namespace vecmem::details

namespace std {

   /// Specialisation of @c std::iterator_traits
   ///
   /// This is necessary to make @c vecmem::reverse_iterator functional on top
   /// of @c vecmem::details::jagged_device_vector_iterator.
   ///
   template< typename T >
   struct
   iterator_traits< vecmem::details::jagged_device_vector_iterator< T > > {
      typedef typename
         vecmem::details::jagged_device_vector_iterator< T >::value_type
         value_type;
      typedef typename
         vecmem::details::jagged_device_vector_iterator< T >::difference_type
         difference_type;
      typedef typename
         vecmem::details::jagged_device_vector_iterator< T >::pointer
         pointer;
      typedef typename
         vecmem::details::jagged_device_vector_iterator< T >::reference
         reference;
   };

} // namespace std

// Include the implementation.
#include "vecmem/containers/impl/jagged_device_vector_iterator.ipp"
