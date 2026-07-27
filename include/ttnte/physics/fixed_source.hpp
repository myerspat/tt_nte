#pragma once

#include <functional>
#include <optional>
#include <torch/extension.h>
#include <vector>

namespace ttnte::physics {

/// @brief Specification for a fixed (static, non-eigenvalue) source term,
/// attached directly to a mesh block (volumetric) or a boundary face
/// (incident flux) before assembly. Holds only plain data -- no `linalg::`
/// types -- so it can live on `mesh::MeshBlock`/`mesh::BoundaryInfo` without
/// introducing a `mesh -> linalg` dependency.
struct FixedSource {
  /// Arbitrary function of physical-space, angle, and energy sample points
  /// (one tensor per axis, matching the assembler's own quadrature/ordinate/
  /// group sample points), evaluated via TT-cross. Used for MMS or any other
  /// source that isn't a uniform isotropic strength.
  std::optional<std::function<torch::Tensor(const std::vector<torch::Tensor>&)>>
    function = std::nullopt;
  /// Per-group isotropic source strength, in physical units (e.g.
  /// neutrons/cm^3/s for a volumetric source). The assembler normalizes by
  /// the angular quadrature's weighting_factor (4*pi in 3-D, 2*pi in 2-D, 2
  /// in 1-D) to get the per-direction value.
  std::optional<torch::Tensor> isotropic_strength = std::nullopt;

  /// @return True if either field is set.
  bool defined() const noexcept
  {
    return function.has_value() || isotropic_strength.has_value();
  }
};

} // namespace ttnte::physics
