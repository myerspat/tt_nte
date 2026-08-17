#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/mesh/mesh_block.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include "ttnte/physics/particle_balance.hpp"
#include <memory>

namespace ttnte::physics {

template<typename BlockType, typename ConfigType>
class DGAssembler {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<DGAssembler>;

protected:
  // =================================================================
  // Protected data
  /// Configuration for assembly.
  ConfigType config_;
  /// Pointer to the mesh block.
  mesh::MeshBlock<BlockType>::Ptr block_;
  /// Pointer to the linear system after assembly.
  linalg::LinearSystem::Ptr linear_system_;

  // =================================================================
  // Protected constructors
  // NOTE: the parameter type must be `const ConfigType&`, not the base
  // `DGAssemblerConfig&` -- config_ has no user-declared constructors, so
  // `config_(config)` with a base-typed `config` invokes C++20's
  // parenthesized-aggregate-init (P0960): it copies only the base subobject
  // and silently DEFAULT-initializes every derived-only field (interior_
  // loss_fmt, ..., assemble_moment_projector, moment_order), discarding
  // whatever the caller actually set on them. Matching the parameter type to
  // ConfigType makes this an ordinary same-type copy, no slicing possible.
  DGAssembler(
    const mesh::MeshBlock<BlockType>::Ptr& block, const ConfigType& config)
    : block_(block), config_(config)
  {}

public:
  virtual ~DGAssembler() = default;

  // =================================================================
  // Public methods
  /// @brief Assemble the linear system for this mesh block.
  /// @return The full linear system object.
  virtual linalg::LinearSystem::Ptr assemble() = 0;
  /// @brief Compute this patch's own particle-balance diagnostics from a
  /// converged state, purely locally (no data shared between patches). This
  /// is declared here -- rather than only on DGFirstOrderTransportAssembler
  /// -- specifically so callers that only know the NumDim-erased
  /// `DGAssembler::Ptr` (e.g. `TransportSolution`, which is not templated on
  /// NumDim) can still call it; see `TransportDriver::get_assemblers()`.
  /// @param psi The converged state to compute the balance from.
  /// @param eps TT-rounding tolerance for the balance/leakage functionals'
  /// own construction, independent of this assembler's own solve tolerance.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return This patch's PatchBalance.
  virtual PatchBalance compute_balance(
    const linalg::State& psi, double eps, int64_t max_rank) = 0;

  // =================================================================
  // Public getters / setters
  /// @return The assembly configuration.
  const DGAssemblerConfig& get_config() const noexcept { return config_; }
  /// @return The mesh block for this assembler.
  const mesh::MeshBlock<BlockType>::Ptr get_block() const noexcept
  {
    return block_;
  }
  /// @return The assembled linear system.
  const linalg::LinearSystem::Ptr& get_linear_system() const noexcept
  {
    return linear_system_;
  }
};

} // namespace ttnte::physics
