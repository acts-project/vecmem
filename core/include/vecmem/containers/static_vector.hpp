/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/types.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <vector>

namespace vecmem {

   /// Class mimicking @c std::vector on top of a fixed sized array
   template< typename TYPE, std::size_t MAX_SIZE >
   class static_vector {

   public:
      /// @name Type definitions mimicking @c std::vector
      /// @{
      typedef std::size_t    size_type;
      typedef std::ptrdiff_t difference_type;
      typedef TYPE*          pointer;
      typedef const TYPE*    const_pointer;
      typedef TYPE&          reference;
      typedef const TYPE&    const_reference;
      typedef TYPE           value_type;
      static const size_type array_max_size = MAX_SIZE;
      typedef TYPE           array_type[ array_max_size ];
      typedef typename std::vector< TYPE >::iterator         iterator;
      typedef typename std::vector< TYPE >::const_iterator   const_iterator;
      typedef typename std::vector< TYPE >::reverse_iterator reverse_iterator;
      typedef typename std::vector< TYPE >::const_reverse_iterator
         const_reverse_iterator;
      /// @}

      /// Default constructor
      VECMEM_HOST_AND_DEVICE
      static_vector( std::size_t size = 0 ) : m_size( size ) {}

      /// @name Vector element access functions
      /// @{
      VECMEM_HOST_AND_DEVICE
      reference at( size_type pos ) {
         assert( pos < m_size );
         return m_elements[ pos ];
      }

      VECMEM_HOST_AND_DEVICE
      const_reference at( size_type pos ) const {
         assert( pos < m_size );
         return m_elements[ pos ];
      }

      VECMEM_HOST_AND_DEVICE
      reference operator[]( size_type pos ) {
         return m_elements[ pos ];
      }
      VECMEM_HOST_AND_DEVICE
      const_reference operator[]( size_type pos ) const {
         return m_elements[ pos ];
      }

      VECMEM_HOST_AND_DEVICE
      reference front() {
         return m_elements[ 0 ];
      }
      VECMEM_HOST_AND_DEVICE
      const_reference front() const {
         return m_elements[ 0 ];
      }

      VECMEM_HOST_AND_DEVICE
      reference back() {
         return m_elements[ m_size - 1 ];
      }
      VECMEM_HOST_AND_DEVICE
      const_reference back() const {
         return m_elements[ m_size - 1 ];
      }
      /// @}

      /// @name Payload modification functions
      /// @{
      VECMEM_HOST_AND_DEVICE
      void push_back( const_reference value ) {
         assert( m_size + 1 <= array_max_size );
         m_elements[ m_size ] = value;
         ++m_size;
      }
      /// @}

      /// @name Iterator providing functions
      /// @{
      VECMEM_HOST
      iterator begin() {
         return iterator( m_elements );
      }
      VECMEM_HOST
      const_iterator begin() const {
         return const_iterator( m_elements );
      }
      VECMEM_HOST
      const_iterator cbegin() const {
         return begin();
      }

      VECMEM_HOST
      iterator end() {
         return iterator( m_elements + m_size );
      }
      VECMEM_HOST
      const_iterator end() const {
         return const_iterator( m_elements + m_size );
      }
      VECMEM_HOST
      const_iterator cend() const {
         return const_iterator( m_elements + m_size );
      }

      VECMEM_HOST
      reverse_iterator rbegin() {
         return reverse_iterator( end() );
      }
      VECMEM_HOST
      const_reverse_iterator rbegin() const {
         return const_reverse_iterator( end() );
      }
      VECMEM_HOST
      const_reverse_iterator crbegin() const {
         return const_reverse_iterator( cend() );
      }

      VECMEM_HOST
      reverse_iterator rend() {
         return reverse_iterator( begin() );
      }
      VECMEM_HOST
      const_reverse_iterator rend() const {
         return const_reverse_iterator( begin() );
      }
      VECMEM_HOST
      const_reverse_iterator crend() const {
         return const_reverse_iterator( cbegin() );
      }
      /// @}

      /// @name Additional helper functions
      /// @{
      VECMEM_HOST_AND_DEVICE
      bool empty() const {
         return m_size == 0;
      }

      VECMEM_HOST_AND_DEVICE
      size_type size() const {
         return m_size;
      }

      VECMEM_HOST_AND_DEVICE
      void resize( std::size_t value ) {
         assert( value <= array_max_size );
         m_size = value;
         return;
      }
      /// @}

   private:
      /// Size of the vector
      size_type m_size;
      /// Array that holds the PoD elements of the vector
      array_type m_elements;

   }; // class static_vector

} // namespace vecmem
