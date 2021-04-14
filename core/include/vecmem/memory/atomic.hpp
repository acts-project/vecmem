/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/types.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {

   /// Atomic access helper class for use in device code
   ///
   /// It is only meant to be used with primitive types. Ones that CUDA, HIP and
   /// SYCL built-in functions exist for. So no structs, or even pointers.
   ///
   /// Note that it is also not meant to work in host code. Support in host
   /// code could be added with @c std::atomic_ref in C++20, but until then
   /// this type will only work in "device code" for atomic access.
   ///
   template< typename T >
   class atomic {

   public:
      /// @name Type definitions
      /// @{

      /// Type managed by the object
      typedef T value_type;
      /// Difference between two objects
      typedef value_type difference_type;
      /// Pointer to the value in global memory
      typedef value_type* pointer;
      /// Reference to a value given by the user
      typedef value_type& reference;

      /// @}

      /// @name Check(s) on the value type
      /// @{

      static_assert( std::is_integral< value_type >::value ,
                     "vecmem::atomic only accepts built-in integral types" );

      /// @}

      /// Constructor, with a pointer to the managed variable
      VECMEM_DEVICE
      atomic( pointer ptr );

      /// @name Value setter/getter functions
      /// @{

      /// Set the variable to the desired value
      VECMEM_DEVICE
      void store( value_type data );
      /// Get the value of the variable
      VECMEM_DEVICE
      value_type load() const;

      /// Exchange the current value of the variable with a different one
      VECMEM_DEVICE
      value_type exchange( value_type data );

      /// Compare against the current value, and exchange only if different
      VECMEM_DEVICE
      bool compare_exchange_strong( reference expected, value_type desired );

      /// @}

      /// @name Value modifier functions
      /// @{

      /// Add a chosen value to the stored variable
      VECMEM_DEVICE
      value_type fetch_add( value_type data );
      /// Substitute a chosen value from the stored variable
      VECMEM_DEVICE
      value_type fetch_sub( value_type data );

      /// Replace the current value with the specified value AND-ed to it
      VECMEM_DEVICE
      value_type fetch_and( value_type data );
      /// Replace the current value with the specified value OR-d to it
      VECMEM_DEVICE
      value_type fetch_or( value_type data );
      /// Replace the current value with the specified value XOR-d to it
      VECMEM_DEVICE
      value_type fetch_xor( value_type data );

      /// @}

   private:
      /// Pointer to the value to perform atomic operations on
      pointer m_ptr;

   }; // class atomic

} // namespace vecmem

// Include the implementation.
#include "vecmem/memory/impl/atomic.ipp"
