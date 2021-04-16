/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// SYCL include(s).
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#   include <CL/sycl/atomic.hpp>
#endif

/// Helpers for super explicit calls to the SYCL atomic functions
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#   define __VECMEM_SYCL_ATOMIC_CALL0( FNAME, PTR )                            \
   cl::sycl::atomic_##FNAME< value_type,                                       \
                             cl::sycl::access::address_space::global_space >(  \
      cl::sycl::atomic< value_type >(                                          \
         cl::sycl::multi_ptr< value_type,                                      \
            cl::sycl::access::address_space::global_space >( PTR ) ) )
#   define __VECMEM_SYCL_ATOMIC_CALL1( FNAME, PTR, ARG1 )                      \
   cl::sycl::atomic_##FNAME< value_type,                                       \
                             cl::sycl::access::address_space::global_space >(  \
      cl::sycl::atomic< value_type >(                                          \
         cl::sycl::multi_ptr< value_type,                                      \
            cl::sycl::access::address_space::global_space >( PTR ) ), ARG1 )
#   define __VECMEM_SYCL_ATOMIC_CALL2( FNAME, PTR, ARG1, ARG2 )                \
   cl::sycl::atomic_##FNAME< value_type,                                       \
                             cl::sycl::access::address_space::global_space >(  \
      cl::sycl::atomic< value_type >(                                          \
         cl::sycl::multi_ptr< value_type,                                      \
            cl::sycl::access::address_space::global_space >( PTR ) ), ARG1,    \
            ARG2 )
#endif

namespace vecmem {

   template< typename T >
   VECMEM_DEVICE
   atomic< T >::atomic( pointer ptr )
   : m_ptr( ptr ) {

   }

   template< typename T >
   VECMEM_DEVICE
   void atomic< T >::store( value_type data ) {

#if defined(__CUDACC__) || defined(__HIPCC__)
      volatile pointer addr = m_ptr;
      __threadfence();
      *addr = data;
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      __VECMEM_SYCL_ATOMIC_CALL1( store, m_ptr, data );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
#endif
   }

   template< typename T >
   VECMEM_DEVICE
   typename atomic< T >::value_type
   atomic< T >::load() const {

#if defined(__CUDACC__) || defined(__HIPCC__)
      volatile pointer addr = m_ptr;
      __threadfence();
      const value_type value = *addr;
      __threadfence();
      return value;
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      return __VECMEM_SYCL_ATOMIC_CALL0( load, m_ptr );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
      return 0;
#endif
   }

   template< typename T >
   VECMEM_DEVICE
   typename atomic< T >::value_type
   atomic< T >::exchange( value_type data ) {

#if defined(__CUDACC__) || defined(__HIPCC__)
      return atomicExch( m_ptr, data );
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      return __VECMEM_SYCL_ATOMIC_CALL1( exchange, m_ptr, data );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
      return 0;
#endif
   }

   template< typename T >
   VECMEM_DEVICE
   bool atomic< T >::compare_exchange_strong( reference expected,
                                              value_type desired ) {

#if defined(__CUDACC__) || defined(__HIPCC__)
      return atomicCAS( m_ptr, expected, desired );
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      return __VECMEM_SYCL_ATOMIC_CALL2( compare_exchange_strong, m_ptr,
                                         expected, desired );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
      return false;
#endif
   }

   template< typename T >
   VECMEM_DEVICE
   typename atomic< T >::value_type
   atomic< T >::fetch_add( value_type data ) {

#if defined(__CUDACC__) || defined(__HIPCC__)
      return atomicAdd( m_ptr, data );
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      return __VECMEM_SYCL_ATOMIC_CALL1( fetch_add, m_ptr, data );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
      return 0;
#endif
   }

   template< typename T >
   VECMEM_DEVICE
   typename atomic< T >::value_type
   atomic< T >::fetch_sub( value_type data ) {

#if defined(__CUDACC__) || defined(__HIPCC__)
      return atomicSub( m_ptr, data );
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      return __VECMEM_SYCL_ATOMIC_CALL1( fetch_sub, m_ptr, data );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
#endif
   }

   template< typename T >
   VECMEM_DEVICE
   typename atomic< T >::value_type
   atomic< T >::fetch_and( value_type data ) {

#if defined(__CUDACC__) || defined(__HIPCC__)
      return atomicAnd( m_ptr, data );
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      return __VECMEM_SYCL_ATOMIC_CALL1( fetch_and, m_ptr, data );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
#endif
   }

   template< typename T >
   VECMEM_DEVICE
   typename atomic< T >::value_type
   atomic< T >::fetch_or( value_type data ) {

#if defined(__CUDACC__) || defined(__HIPCC__)
      return atomicOr( m_ptr, data );
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      return __VECMEM_SYCL_ATOMIC_CALL1( fetch_or, m_ptr, data );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
#endif
   }

   template< typename T >
   VECMEM_DEVICE
   typename atomic< T >::value_type
   atomic< T >::fetch_xor( value_type data ) {

#if defined(__CUDACC__) || defined(__HIPCC__)
      return atomicXor( m_ptr, data );
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
      return __VECMEM_SYCL_ATOMIC_CALL1( fetch_xor, m_ptr, data );
#else
      static_assert( false, "vecmem::atomic is not meant for host code" );
#endif
   }

} // namespace vecmem