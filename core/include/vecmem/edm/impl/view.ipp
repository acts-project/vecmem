/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/device_traits.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/utils/type_traits.hpp"

// System include(s).
#include <cassert>

namespace vecmem {
namespace edm {

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE view<schema<VARTYPES...>, INTERFACE>::view(
    size_type capacity, const memory_view_type& size)
    : m_capacity(capacity),
      m_views{},
      m_size{size},
      m_payload{0, nullptr},
      m_layout{0, nullptr},
      m_host_layout{0, nullptr} {}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <typename... OTHERTYPES, typename>
VECMEM_HOST_AND_DEVICE view<schema<VARTYPES...>, INTERFACE>::view(
    const view<schema<OTHERTYPES...>, INTERFACE>& other)
    : m_capacity{other.capacity()},
      m_views{other.variables()},
      m_size{other.view_size()},
      m_payload{other.payload()},
      m_layout{other.layout()},
      m_host_layout{other.host_layout()} {}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <typename... OTHERTYPES, typename>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::operator=(
    const view<schema<OTHERTYPES...>, INTERFACE>& rhs) -> view& {

    // Note that self-assignment with this function should never be a thing.
    // So we don't need to check for it in production code.
    assert(static_cast<const void*>(this) != static_cast<const void*>(&rhs));

    // Copy the data from the other view.
    m_capacity = rhs.capacity();
    m_views = rhs.variables();
    m_size = rhs.view_size();
    m_payload = rhs.payload();
    m_layout = rhs.layout();
    m_host_layout = rhs.host_layout();

    // Return a reference to this object.
    return *this;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::size() const
    -> size_type {

    const_size_pointer size_ptr =
        details::device_size_pointer<vecmem::details::disjunction_v<
            type::details::is_jagged_vector<VARTYPES>...>>::get(m_size);
    return (size_ptr == nullptr ? m_capacity : *size_ptr);
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::capacity()
    const -> size_type {

    return m_capacity;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto
view<schema<VARTYPES...>, INTERFACE>::push_back_default() -> size_type {

    // There must be no jagged vector variables for this to work.
    static_assert(!vecmem::details::disjunction<
                      type::details::is_jagged_vector<VARTYPES>...>::value,
                  "Containers with jagged vector variables cannot be resized!");
    // There must be at least one vector variable in the container.
    static_assert(vecmem::details::disjunction<
                      type::details::is_vector<VARTYPES>...>::value,
                  "This function requires at least one vector variable.");

    // This can only be done on a resizable container.
    size_pointer size_ptr =
        details::device_size_pointer<vecmem::details::disjunction_v<
            type::details::is_jagged_vector<VARTYPES>...>>::get(m_size);
    assert(size_ptr != nullptr);

    // Increment the size of the container at first. So that we would "claim"
    // the index from other threads.
    device_atomic_ref<size_type> asize(*size_ptr);
    const size_type index = asize.fetch_add(1u);
    assert(index < m_capacity);

    // Construct the new elements in all of the vector variables.
    construct_default(index, std::index_sequence_for<VARTYPES...>{});

    // Return the position of the new variable(s).
    return index;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::push_back(
    const object_type& element) -> size_type {

    // Add a new default element to the container.
    const size_type index = push_back_default();

    // Set it to the given value.
    at(index) = element;

    // Return the position of the new variable(s).
    return index;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE
    typename details::view_type_at<INDEX, VARTYPES...>::return_type
    view<schema<VARTYPES...>, INTERFACE>::get() {

    return details::view_get<tuple_element_t<INDEX, tuple<VARTYPES...>>>::get(
        vecmem::get<INDEX>(m_views));
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE
    typename details::view_type_at<INDEX, VARTYPES...>::const_return_type
    view<schema<VARTYPES...>, INTERFACE>::get() const {

    return details::view_get<tuple_element_t<INDEX, tuple<VARTYPES...>>>::get(
        vecmem::get<INDEX>(m_views));
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE void view<schema<VARTYPES...>, INTERFACE>::set(
    std::add_lvalue_reference_t<std::add_const_t<
        typename details::view_type_at<INDEX, VARTYPES...>::type>>
        value) {

    vecmem::get<INDEX>(m_views) = value;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::at(
    size_type index) -> proxy_type {

    // Make sure that the index is within bounds.
    assert(index < size());

    // Use the unprotected function.
    return this->operator[](index);
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::at(
    size_type index) const -> const_proxy_type {

    // Make sure that the index is within bounds.
    assert(index < size());

    // Use the unprotected function.
    return this->operator[](index);
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::operator[](
    size_type index) -> proxy_type {

    // Create the proxy.
    return proxy_type{*this, index};
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::operator[](
    size_type index) const -> const_proxy_type {

    // Create the proxy.
    return const_proxy_type{*this, index};
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::variables()
    -> tuple_type& {

    return m_views;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::variables()
    const -> const tuple_type& {

    return m_views;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::view_size()
    const -> const memory_view_type& {

    return m_size;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::payload()
    const -> const memory_view_type& {

    return m_payload;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::layout() const
    -> const memory_view_type& {

    return m_layout;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>, INTERFACE>::host_layout()
    const -> const memory_view_type& {

    return m_host_layout;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <std::size_t INDEX, std::size_t... Is>
VECMEM_HOST_AND_DEVICE void
view<schema<VARTYPES...>, INTERFACE>::construct_default(
    size_type index, std::index_sequence<INDEX, Is...>) const {

    // Construct the new element in this variable, if it's a vector.
    construct_vector(index, this->template get<INDEX>());
    // Continue the recursion.
    construct_default(index, std::index_sequence<Is...>{});
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST_AND_DEVICE void
view<schema<VARTYPES...>, INTERFACE>::construct_default(
    size_type, std::index_sequence<>) const {
    // The terminal node doesn't need to do anything.
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <typename T>
VECMEM_HOST_AND_DEVICE void
view<schema<VARTYPES...>, INTERFACE>::construct_vector(size_type, T&) const {
    // Nothing is done for non-vector variables.
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <typename T>
VECMEM_HOST_AND_DEVICE void
view<schema<VARTYPES...>, INTERFACE>::construct_vector(
    size_type index, vecmem::data::vector_view<T>& vec) const {

    vec.construct(index, {});
}

namespace details {

template <std::size_t INDEX, template <typename> class INTERFACE,
          typename... VARTYPES>
struct get_capacities_impl {

    VECMEM_HOST
    static std::vector<vecmem::data::vector_view<int>::size_type> get(
        const view<schema<VARTYPES...>, INTERFACE>& soa) {

        return get_impl(soa, soa.template get<INDEX>());
    }

private:
    template <typename T>
    VECMEM_HOST static std::vector<vecmem::data::vector_view<int>::size_type>
    get_impl(const view<schema<VARTYPES...>, INTERFACE>& soa, const T&) {

        return get_capacities_impl<INDEX - 1, INTERFACE, VARTYPES...>::get(soa);
    }

    template <typename T>
    VECMEM_HOST static std::vector<vecmem::data::vector_view<int>::size_type>
    get_impl(const view<schema<VARTYPES...>, INTERFACE>& soa,
             const vecmem::data::jagged_vector_view<T>& vec) {

        auto this_size = vecmem::data::get_capacities(vec);
        auto other_size =
            get_capacities_impl<INDEX - 1, INTERFACE, VARTYPES...>::get(soa);
        assert(this_size.size() == other_size.size());
        std::vector<vecmem::data::vector_view<int>::size_type> result(
            soa.capacity());
        assert(result.size() == this_size.size());
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = std::max(this_size[i], other_size[i]);
        }
        return result;
    }

};  // struct get_capacities_impl

template <template <typename> class INTERFACE, typename... VARTYPES>
struct get_capacities_impl<0u, INTERFACE, VARTYPES...> {

    VECMEM_HOST
    static std::vector<vecmem::data::vector_view<int>::size_type> get(
        const view<schema<VARTYPES...>, INTERFACE>& soa) {

        return get_impl(soa, soa.template get<0u>());
    }

private:
    template <typename T>
    VECMEM_HOST static std::vector<vecmem::data::vector_view<int>::size_type>
    get_impl(const view<schema<VARTYPES...>, INTERFACE>& soa, const T&) {

        return std::vector<vecmem::data::vector_view<int>::size_type>(
            soa.capacity(), 0);
    }

    template <typename T>
    VECMEM_HOST static std::vector<vecmem::data::vector_view<int>::size_type>
    get_impl(const view<schema<VARTYPES...>, INTERFACE>&,
             const vecmem::data::jagged_vector_view<T>& vec) {

        return vecmem::data::get_capacities(vec);
    }
};  // struct get_capacities_impl

}  // namespace details

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST std::vector<vecmem::data::vector_view<int>::size_type>
get_capacities(const view<schema<VARTYPES...>, INTERFACE>& soa) {

    // Make sure that there's a jagged vector in here.
    static_assert(
        details::has_jagged_vector<schema<VARTYPES...>>::value,
        "Function can only be used on containers with jagged vectors");

    return details::get_capacities_impl<sizeof...(VARTYPES) - 1, INTERFACE,
                                        VARTYPES...>::get(soa);
}

}  // namespace edm
}  // namespace vecmem
