// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-hermite.h"
#include "element-families.h"
#include "lattice.h"
#include "maps.h"
#include "math.h"
#include "mdspan.hpp"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <array>
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> basix::element::create_hct(cell::type celltype, int degree,
                                                bool discontinuous)
{
  if (celltype != cell::type::triangle)
  {
    throw std::runtime_error("Unsupported cell type");
  }

  if (degree != 3)
    throw std::runtime_error("Hsieh-Clough-Tocher element must have degree 3");

  // Only triangle and each edge contains one dof.
  const std::size_t tdim = 2;
  const std::size_t ndofs = 12;
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  const auto [gdata, gshape] = cell::geometry<T>(celltype);
  impl::mdspan_t<const T, 2> geometry(gdata.data(), gshape);
  const std::size_t deriv_count = polyset::nderivs(celltype, 1);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  // Vertices -- 3 * 3 dofs
  for (std::size_t e = 0; e < topology[0].size(); ++e)
  {
    const auto [entity_x, entity_shape]
        = cell::sub_entity_geometry<T>(celltype, 0, e);
    x[0].emplace_back(entity_shape, entity_x);
    auto& _M = M[0].emplace_back(1 + tdim, 1, 1, deriv_count);
    _M(0, 0, 0, 0) = 1;
    for (std::size_t d = 0; d < tdim; ++d)
      _M(d + 1, 0, 0, d + 1) = 1;
  }

  // Edges -- 3 * 1 dofs
  const auto facet_normols = cell::facet_normals<T>(celltype);
  for (std::size_t e = 0; e < topology[1].size(); ++e)
  {
    x[1].emplace_back(0, tdim);
    M[1].emplace_back(0, 1, 0, deriv_count);
  }

  // Facets -- 0 dofs
  for (std::size_t e = 0; e < topology[2].size(); ++e)
  {
    x[1].emplace_back(0, tdim);
    M[1].emplace_back(0, 1, 0, deriv_count);
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, 1);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H2;
  return FiniteElement<T>(
      element::family::HCT, celltype, polyset::type::standard, degree, {},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs), xview,
      Mview, 1, maps::type::identity, space, discontinuous, -1, degree,
      element::lagrange_variant::unset, element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template FiniteElement<float> element::create_hct(cell::type, int, bool);
template FiniteElement<double> element::create_hct(cell::type, int, bool);
//-----------------------------------------------------------------------------
