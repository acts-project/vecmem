/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/const_device_vector_data.hpp"
#include "vecmem/utils/reverse_iterator.hpp"
#include "vecmem/utils/types.hpp"

namespace vecmem {

   /// Class mimicking a constant @c std::vector in "device code"
   ///
   /// This type can be used in "generic device code" as a constant
   /// @c std::vector. It does not allow the vector or its elements to be
   /// modified, it just allows client code to access the data wrapped by the
   /// vector with the same interface that @c std::vector provides.
   ///
   template< typename TYPE >
   class const_device_vector {

   public:
      /// @name Type definitions, mimicking @c std::vector
      /// @{

      /// Type of the array elements
      typedef TYPE              value_type;
      /// Size type for the array
      typedef std::size_t       size_type;
      /// Pointer difference type
      typedef std::ptrdiff_t    difference_type;

      /// Constant value reference type
      typedef const value_type& const_reference;
      /// Constant value pointer type
      typedef const value_type* const_pointer;

      /// Constant forward iterator type
      typedef const_pointer     const_iterator;
      /// Constant reverse iterator type
      typedef reverse_iterator< const_iterator > const_reverse_iterator;

      /// @}

      /// Constructor, on top of a previously allocated/filled block of memory
      VECMEM_HOST_AND_DEVICE
      const_device_vector( const const_device_vector_data< value_type >& data );
      /// Copy constructor
      VECMEM_HOST_AND_DEVICE
      const_device_vector( const const_device_vector& parent );

      /// Copy assignment operator
      VECMEM_HOST_AND_DEVICE
      const_device_vector& operator=( const const_device_vector& rhs );

      /// @name Vector element access functions
      /// @{

      /// Return a specific element of the vector in a "safe/checked way"
      VECMEM_HOST_AND_DEVICE
      const_reference at( size_type pos ) const;
      /// Return a specific element of the vector
      VECMEM_HOST_AND_DEVICE
      const_reference operator[]( size_type pos ) const;
      /// Return the first element of the vector.
      VECMEM_HOST_AND_DEVICE
      const_reference front() const;
      /// Return the last element of the vector
      VECMEM_HOST_AND_DEVICE
      const_reference back() const;
      /// Access the underlying memory array
      VECMEM_HOST_AND_DEVICE
      const_pointer data() const;

      /// @}

      /// @name Iterator providing functions
      /// @{

      /// Return a forward iterator pointing at the beginning of the vector
      VECMEM_HOST_AND_DEVICE
      const_iterator begin() const;
      /// Return a forward iterator pointing at the beginning of the vector
      VECMEM_HOST_AND_DEVICE
      const_iterator cbegin() const;

      /// Return a forward iterator pointing at the end of the vector
      VECMEM_HOST_AND_DEVICE
      const_iterator end() const;
      /// Return a forward iterator pointing at the end of the vector
      VECMEM_HOST_AND_DEVICE
      const_iterator cend() const;

      /// Return a reverse iterator pointing at the end of the vector
      VECMEM_HOST_AND_DEVICE
      const_reverse_iterator rbegin() const;
      /// Return a reverse iterator pointing at the end of the vector
      VECMEM_HOST_AND_DEVICE
      const_reverse_iterator crbegin() const;

      /// Return a reverse iterator pointing at the beginning of the vector
      VECMEM_HOST_AND_DEVICE
      const_reverse_iterator rend() const;
      /// Return a reverse iterator pointing at the beginning of the vector
      VECMEM_HOST_AND_DEVICE
      const_reverse_iterator crend() const;

      /// @}

      /// @name Capacity checking functions
      /// @{

      /// Check whether the vector is empty
      VECMEM_HOST_AND_DEVICE
      bool empty() const;
      /// Return the number of elements in the vector
      VECMEM_HOST_AND_DEVICE
      size_type size() const;
      /// Return the maximum (fixed) number of elements in the vector
      VECMEM_HOST_AND_DEVICE
      size_type max_size() const;
      /// Return the current (fixed) capacity of the vector
      VECMEM_HOST_AND_DEVICE
      size_type capacity() const;

      /// @}

   private:
      /// Size of the array that this object looks at
      size_type m_size;
      /// Pointer to the start of the array
      const_pointer m_ptr;

   }; // class const_device_vector

} // namespace vecmem

// Include the implementation.
#include "vecmem/containers/const_device_vector.ipp"
