/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include <memory>

#include "vecmem/utils/abstract_event.hpp"
#include "vecmem/vecmem_cuda_export.hpp"

namespace vecmem::cuda {

class VECMEM_CUDA_EXPORT event_pool {
public:
    using event_type = std::unique_ptr<abstract_event>;

    explicit event_pool(std::size_t _n);

    event_pool();

    ~event_pool();

    event_type create_event() const;

    void free_event(void *) const;

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace vecmem::cuda
