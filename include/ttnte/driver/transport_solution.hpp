#pragma once

#include "ttnte/cad/patch.hpp"
#include "ttnte/linalg/ops.hpp"
#include "ttnte/linalg/state.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include "ttnte/physics/dg_assembler.hpp"
#include "ttnte/physics/dg_first_order_transport_assembler.hpp"
#include "ttnte/physics/dg_first_order_transport_backends.hpp"
#include "ttnte/physics/particle_balance.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/label.hpp"
#include "ttnte/xs/server.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ttnte::driver {

/// @brief A distributed, per-patch linalg::State container -- keyed by GID,
/// holding only this rank's own local patches -- with the machinery to
/// post-process a solved TransportDriver result. This class does not know
/// what physical quantity it holds: the raw angular flux
/// (TransportDriver::get_solution()) and any derived field (e.g. scalar flux
/// via compute_scalar_flux()) are both just a TransportSolution holding a
/// different State per patch.
template<typename BlockType>
class TransportSolution {
public:
  // =================================================================
  // Public types
  using Mesh = mesh::Mesh<BlockType>;
  using Label = utils::Label<TransportSolution>;
  using Ptr = std::shared_ptr<TransportSolution>;
  /// NumDim-erased assembler handle -- TransportSolution isn't templated on
  /// NumDim (unlike TransportDriver/its Assembler type), so
  /// compute_patch_balances()/patch_balance_table()/global_balance() accept
  /// this base-class Ptr instead (see TransportDriver::get_assemblers()).
  using AssemblerBase =
    physics::DGAssembler<BlockType, physics::DGTransportAssemblerConfig>;
  using AssemblerPtr = typename AssemblerBase::Ptr;

private:
  /// Per-face header width in pack_patch_balance()'s flat row layout: dim,
  /// is_upper, type, neighbor_gid, neighbor_dim, neighbor_is_upper,
  /// has_incoming.
  static constexpr size_t BALANCE_FACE_HEADER_WIDTH = 7;

  // =================================================================
  // Private data
  Label label_;
  /// The converged k-eigenvalue, if this solution came from an eigenvalue
  /// solve. Unset for solve modes that don't produce one (e.g. fixed-source).
  std::optional<double> k_eff_;
  /// GID -> this rank's local field. Only contains entries for GIDs local to
  /// this rank.
  std::unordered_map<int64_t, linalg::State> local_fields_;
  /// The angular quadrature set used to assemble the originating solve, for
  /// compute_scalar_flux(). Null if unavailable (e.g. assemble() was never
  /// called with a quadrature set).
  math::QuadratureSet::Ptr angular_qset_;
  /// This rank's own (already-culled, already-local) mesh -- referenced, not
  /// copied, for local-only geometry access (e.g. regular_mesh_average(),
  /// gather_plot_data()). No non-local geometry is ever held here.
  typename Mesh::Ptr mesh_;
  /// The XS server used to assemble the originating solve, retained (like
  /// angular_qset_) so make_assembler() can build a fresh assembler on
  /// demand for compute_patch_balances()/patch_balance_table()/
  /// global_balance() when the caller doesn't already have one.
  xs::Server::Ptr xs_server_;
  /// The assembler config used to assemble the originating solve, retained
  /// for the same reason as xs_server_ -- a fresh assembler must be rounded
  /// at the ORIGINAL solve's tolerance so its source_ matches what the real
  /// solve actually saw.
  physics::DGTransportAssemblerConfig config_;
  /// Communicator for MPI reductions/gathers.
  parallel::Communicator comm_;
  /// Whether local_fields_ still carries angular dependence (true for what
  /// TransportDriver::solve_eigenvalue() returns) or has already been
  /// reduced (false for compute_scalar_flux()'s result). Tracked explicitly
  /// rather than inferred from tensor rank after the fact.
  bool has_angular_dependence_ = true;

  // =================================================================
  // Private methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::driver::TransportSolution::" + func_name;
  }

  /// @brief Recompute the shared spatial DOF-to-quadrature-point evaluation
  /// maps and Jacobian-weighted quadrature weight needed to compare `block`'s
  /// field against `ref_block`'s. Cheap (purely geometric, no
  /// cross-section/material dependency -- see the material-free
  /// DGFirstOrderTransportBackend constructors), so not cached; used only by
  /// compute_errors().
  ///
  /// `block` and `ref_block` are assumed to share the same parametric domain
  /// (e.g. `ref_block` is a differently-refined -- more knot spans and/or
  /// higher degree -- version of the same geometry), so no point inversion
  /// is needed. But the two sides' own auto-derived quadratures are each
  /// only accurate enough to integrate THEIR OWN degree/knot spans exactly
  /// -- using the coarser side's quadrature to evaluate the finer side's
  /// basis would under-integrate it. So this picks, WHOLESALE (one winner
  /// for the whole patch), whichever side has more total quadrature points
  /// summed across dimensions, and uses that side's own quadrature AND
  /// Jacobian mapping as the shared integration measure; the coarser side's
  /// basis is then evaluated at the finer side's quadrature points (a
  /// coarser/lower-degree basis is still integrated exactly by a rule built
  /// for a finer one -- the reverse is not true).
  /// @return {basis for `block`, shared Jacobian-weighted mapping, basis for
  /// `ref_block`} -- all three evaluated at the SAME (finer side's)
  /// quadrature points.
  template<int64_t NumDim, linalg::FormatType fmt>
  static std::tuple<linalg::TTEngine, linalg::TTEngine, linalg::TTEngine>
  compute_error_weights_impl(const typename Mesh::BlockTypePtr& block,
    const typename Mesh::BlockTypePtr& ref_block,
    const math::QuadratureSet::Ptr& angular_qset)
  {
    auto config = physics::DGTransportAssemblerConfig();
    config.rounding.eps = 0;
    config.rounding.max_rank = std::numeric_limits<int>::max();
    config.cross_jacobian_inverse = false;

    using Backend =
      physics::backends::DGFirstOrderTransportBackend<BlockType, fmt, NumDim>;

    auto total_quad_points = [](const auto& points) {
      int64_t total = 0;
      for (const auto& p : points) {
        total += p.numel();
      }
      return total;
    };

    // Each side's own auto-derived quadrature (sized to exactly integrate
    // that side's own degree/knot spans).
    Backend field_backend(block, angular_qset, config);
    Backend ref_backend(ref_block, angular_qset, config);
    bool ref_is_finer = total_quad_points(ref_backend.get_quad_points()) >=
                        total_quad_points(field_backend.get_quad_points());

    Backend& fine_backend = ref_is_finer ? ref_backend : field_backend;
    const auto& coarse_block = ref_is_finer ? block : ref_block;

    linalg::TTEngine fine_basis = fine_backend.assemble_basis();
    linalg::TTEngine mapping = fine_backend.assemble_integral_mapping();

    // Evaluate the coarser side's basis at the finer side's own quadrature
    // points.
    Backend coarse_backend(
      coarse_block, angular_qset, fine_backend.get_quad_points(), config);
    linalg::TTEngine coarse_basis = coarse_backend.assemble_basis();

    linalg::TTEngine basis = ref_is_finer ? coarse_basis : fine_basis;
    linalg::TTEngine ref_basis = ref_is_finer ? fine_basis : coarse_basis;

    return {std::move(basis), std::move(mapping), std::move(ref_basis)};
  }

  template<linalg::FormatType fmt>
  std::tuple<linalg::TTEngine, linalg::TTEngine, linalg::TTEngine>
  compute_error_weights(const typename Mesh::BlockTypePtr& block,
    const typename Mesh::BlockTypePtr& ref_block) const
  {
    switch (block->get_ndim()) {
    case 1:
      return compute_error_weights_impl<1, fmt>(
        block, ref_block, angular_qset_);
    case 2:
      return compute_error_weights_impl<2, fmt>(
        block, ref_block, angular_qset_);
    case 3:
      return compute_error_weights_impl<3, fmt>(
        block, ref_block, angular_qset_);
    default:
      throw utils::runtime_error(*this, error_context("compute_error_weights"),
        "Unsupported patch dimensionality: " +
          std::to_string(block->get_ndim()));
    }
  }

  /// @brief Evaluate a spatial(+energy) field -- e.g. this solution's own
  /// local field for `block`'s GID -- at a batch of scattered PARAMETRIC
  /// points within `block`, via Patch::evaluate_field(). Used only by
  /// regular_mesh_average().
  /// @param block The patch whose basis the field lives on.
  /// @param field This block's own spatial(+energy) field -- NOT angular
  /// (caller must have already reduced via compute_scalar_flux()).
  /// @param points Parametric coordinates, shape (n, ndim).
  /// @return Field values at `points`, shape (n, num_groups).
  torch::Tensor evaluate_field_at_points(
    const typename Mesh::BlockTypePtr& block, const linalg::State& field,
    const torch::Tensor& points) const
  {
    int64_t patch_ndim = block->get_ndim();
    torch::Tensor dense = field.to_dense();

    // Drop the trailing all-1 n-block to_dense() carries for a State,
    // leaving just (spatial_dims..., num_groups).
    c10::SmallVector<int64_t, 6> spatial_shape(
      dense.sizes().begin(), dense.sizes().begin() + patch_ndim + 1);
    torch::Tensor reshaped = dense.reshape(spatial_shape);

    return block->evaluate_field(reshaped, points);
  }

  /// @brief Every GID in the whole mesh, sorted -- the index into
  /// patch_balance_table()/global_balance()'s packed reduction buffer.
  /// mesh_->get_gid2rank() is global metadata, so this is computable on
  /// every rank without any communication, even one that owns zero local
  /// patches.
  std::vector<int64_t> all_gids_sorted() const
  {
    std::vector<int64_t> gids;
    gids.reserve(mesh_->get_gid2rank().size());
    for (const auto& [gid, rank] : mesh_->get_gid2rank()) {
      gids.push_back(gid);
    }
    std::sort(gids.begin(), gids.end());
    return gids;
  }

  /// @brief Pack one patch's PatchBalance into a flat row of doubles, so
  /// every rank's own local patches can be combined into one
  /// globally-consistent table via a single iallreduce(SUM) instead of a
  /// variable-size gatherv (the codebase's existing pattern for
  /// variable-count cross-rank data, e.g. ttnte.mesh._gather.
  /// gather_mesh_patches(), isn't needed here: num_groups and the per-patch
  /// face count are uniform across the whole mesh -- NumDim is one
  /// compile-time value for the whole originating TransportDriver -- and
  /// exactly one rank owns, and therefore contributes a nonzero row for,
  /// each GID; every other rank's row for that GID is all zero).
  /// Row layout: [fixed_source(G) | fission_source(G) | absorption(G) |
  /// scatter_in(G) | scatter_out(G) | leakage(G) | face_0 | face_1 | ...],
  /// each face block being [dim, is_upper, type, neighbor_gid (-1 if
  /// unset), neighbor_dim (-1 if unset), neighbor_is_upper (0/1, undefined
  /// if unset), has_incoming (0/1), outgoing(G), incoming(G) (0-filled if
  /// unset)] -- BALANCE_FACE_HEADER_WIDTH + 2*G doubles per face.
  static void pack_patch_balance(const physics::PatchBalance& pb,
    int64_t num_groups, size_t num_faces, double* row)
  {
    size_t g = static_cast<size_t>(num_groups);
    auto put = [&](const torch::Tensor& t, size_t offset) {
      torch::Tensor flat = t.to(torch::kFloat64).contiguous().reshape({-1});
      if (static_cast<size_t>(flat.numel()) != g) {
        throw utils::runtime_error(
          "ttnte::driver::TransportSolution::pack_patch_balance",
          "A balance tensor has " + std::to_string(flat.numel()) +
            " entries, expected exactly num_groups (" + std::to_string(g) +
            ")");
      }
      std::memcpy(row + offset, flat.data_ptr<double>(), sizeof(double) * g);
    };

    put(pb.fixed_source, 0 * g);
    put(pb.fission_source, 1 * g);
    put(pb.absorption, 2 * g);
    put(pb.scatter_in, 3 * g);
    put(pb.scatter_out, 4 * g);
    put(pb.leakage, 5 * g);

    size_t face_width = BALANCE_FACE_HEADER_WIDTH + 2 * g;
    size_t offset = 6 * g;
    for (size_t f = 0; f < num_faces; f++) {
      const auto& face = pb.faces[f];
      row[offset + 0] = static_cast<double>(face.dim);
      row[offset + 1] = face.is_upper ? 1.0 : 0.0;
      row[offset + 2] = static_cast<double>(face.type);
      row[offset + 3] = face.neighbor_gid.has_value()
                          ? static_cast<double>(*face.neighbor_gid)
                          : -1.0;
      row[offset + 4] = face.neighbor_dim.has_value()
                          ? static_cast<double>(*face.neighbor_dim)
                          : -1.0;
      row[offset + 5] =
        face.neighbor_is_upper.has_value() && *face.neighbor_is_upper ? 1.0
                                                                      : 0.0;
      row[offset + 6] = face.incoming.has_value() ? 1.0 : 0.0;
      put(face.outgoing, offset + BALANCE_FACE_HEADER_WIDTH);
      if (face.incoming.has_value()) {
        put(*face.incoming, offset + BALANCE_FACE_HEADER_WIDTH + g);
      } else {
        std::memset(
          row + offset + BALANCE_FACE_HEADER_WIDTH + g, 0, sizeof(double) * g);
      }
      offset += face_width;
    }
  }

  /// @brief Inverse of pack_patch_balance() -- reconstruct one patch's
  /// PatchBalance from its packed row.
  static physics::PatchBalance unpack_patch_balance(int64_t gid,
    const double* row, int64_t num_groups, size_t num_faces,
    const torch::TensorOptions& options)
  {
    size_t g = static_cast<size_t>(num_groups);
    auto get = [&](size_t offset) {
      torch::Tensor t = torch::from_blob(
        const_cast<double*>(row + offset), {num_groups}, torch::kFloat64);
      return t.clone().to(options);
    };

    physics::PatchBalance pb;
    pb.gid = gid;
    pb.fixed_source = get(0 * g);
    pb.fission_source = get(1 * g);
    pb.absorption = get(2 * g);
    pb.scatter_in = get(3 * g);
    pb.scatter_out = get(4 * g);
    pb.leakage = get(5 * g);

    size_t face_width = BALANCE_FACE_HEADER_WIDTH + 2 * g;
    size_t offset = 6 * g;
    pb.faces.reserve(num_faces);
    for (size_t f = 0; f < num_faces; f++) {
      physics::FaceBalance face;
      face.dim = static_cast<size_t>(std::llround(row[offset + 0]));
      face.is_upper = row[offset + 1] != 0.0;
      face.type = static_cast<physics::BoundaryType>(
        static_cast<int>(std::llround(row[offset + 2])));
      double ngid = row[offset + 3];
      face.neighbor_gid =
        ngid >= 0.0
          ? std::optional<int64_t>(static_cast<int64_t>(std::llround(ngid)))
          : std::nullopt;
      double ndim = row[offset + 4];
      face.neighbor_dim =
        ndim >= 0.0
          ? std::optional<size_t>(static_cast<size_t>(std::llround(ndim)))
          : std::nullopt;
      face.neighbor_is_upper = face.neighbor_gid.has_value()
                                 ? std::optional<bool>(row[offset + 5] != 0.0)
                                 : std::nullopt;
      bool has_incoming = row[offset + 6] != 0.0;
      face.outgoing = get(offset + BALANCE_FACE_HEADER_WIDTH);
      face.incoming = has_incoming ? std::optional<torch::Tensor>(get(
                                       offset + BALANCE_FACE_HEADER_WIDTH + g))
                                   : std::nullopt;
      pb.faces.push_back(std::move(face));
      offset += face_width;
    }
    return pb;
  }

  /// @brief Build a fresh assembler for `block`, dispatching on its NumDim
  /// (mirrors compute_error_weights()'s own get_ndim() switch). This is
  /// compute_patch_balances()'s fallback for any GID whose assembler wasn't
  /// passed in -- e.g. clear_assemblers=true was used, or the caller simply
  /// doesn't want to keep TransportDriver::get_assemblers() around. assemble()
  /// is called once (the same, already-tested path TransportDriver::assemble()
  /// itself uses) so this assembler's source_ -- needed for fixed_source --
  /// ends up rounded at the SAME tolerance (config_) the original solve
  /// actually used, not a fresh/default one; the resulting LinearSystem is
  /// discarded (only the assembler and its lazily-cached backends are kept).
  AssemblerPtr make_assembler(const typename Mesh::BlockTypePtr& block) const
  {
    AssemblerPtr assembler;
    switch (block->get_ndim()) {
    case 1:
      assembler = physics::DGFirstOrderTransportAssembler<BlockType, 1>::create(
        block, angular_qset_, xs_server_, config_);
      break;
    case 2:
      assembler = physics::DGFirstOrderTransportAssembler<BlockType, 2>::create(
        block, angular_qset_, xs_server_, config_);
      break;
    case 3:
      assembler = physics::DGFirstOrderTransportAssembler<BlockType, 3>::create(
        block, angular_qset_, xs_server_, config_);
      break;
    default:
      throw utils::runtime_error(*this, error_context("make_assembler"),
        "Unsupported patch dimensionality: " +
          std::to_string(block->get_ndim()));
    }
    assembler->assemble();
    return assembler;
  }

  /// @brief make_assembler(), looked up by GID among this rank's own local
  /// mesh blocks.
  AssemblerPtr make_assembler_for_gid(int64_t gid) const
  {
    for (const auto& block : mesh_->get_blocks()) {
      if (block->get_gid() == gid) {
        return make_assembler(block);
      }
    }
    throw utils::runtime_error(*this, error_context("make_assembler_for_gid"),
      "No local mesh block found for GID " + std::to_string(gid));
  }

  /// @brief Collective: combine every rank's own local PatchBalances (from
  /// compute_patch_balances()) into the full, GID-sorted table, identical on
  /// every rank, via one small iallreduce(SUM) -- see pack_patch_balance()
  /// for why a fixed-size reduction suffices here instead of a
  /// variable-size gatherv.
  std::vector<physics::PatchBalance> gather_patch_balances(
    const std::unordered_map<int64_t, physics::PatchBalance>& local) const
  {
    std::vector<int64_t> gids = all_gids_sorted();
    if (gids.empty()) {
      throw utils::runtime_error(*this, error_context("gather_patch_balances"),
        "No patches found in the mesh");
    }

    // num_groups/num_faces aren't knowable locally on a rank that owns zero
    // local patches -- agree on both first via a tiny iallreduce(MAX)
    // (mirrors regular_mesh_average()'s own num_groups handshake).
    int64_t local_header[2] = {0, 0};
    if (!local.empty()) {
      const auto& any_pb = local.begin()->second;
      local_header[0] = any_pb.fixed_source.numel();
      local_header[1] = static_cast<int64_t>(any_pb.faces.size());
    }
    int64_t global_header[2] = {local_header[0], local_header[1]};
    if (comm_.size() > 1) {
      comm_.iallreduce(local_header, global_header, 2, parallel::MPIOp::MAX)
        .wait();
    }
    int64_t num_groups = global_header[0];
    size_t num_faces = static_cast<size_t>(global_header[1]);

    size_t row_width = 6 * static_cast<size_t>(num_groups) +
                       num_faces * (BALANCE_FACE_HEADER_WIDTH +
                                     2 * static_cast<size_t>(num_groups));
    size_t total = gids.size() * row_width;

    std::unordered_map<int64_t, size_t> gid_to_row;
    gid_to_row.reserve(gids.size());
    for (size_t i = 0; i < gids.size(); i++) {
      gid_to_row.emplace(gids[i], i);
    }

    std::vector<double> send(total, 0.0);
    for (const auto& [gid, pb] : local) {
      pack_patch_balance(pb, num_groups, num_faces,
        send.data() + gid_to_row.at(gid) * row_width);
    }

    std::vector<double> recv(total, 0.0);
    if (comm_.size() > 1) {
      comm_
        .iallreduce(send.data(), recv.data(), static_cast<int>(total),
          parallel::MPIOp::SUM)
        .wait();
    } else {
      recv = std::move(send);
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    std::vector<physics::PatchBalance> table;
    table.reserve(gids.size());
    for (size_t i = 0; i < gids.size(); i++) {
      table.push_back(unpack_patch_balance(
        gids[i], recv.data() + i * row_width, num_groups, num_faces, options));
    }
    return table;
  }

  /// @brief Resolve every `INTERNAL` face's `incoming` -- the neighbor's own
  /// `outgoing` through the same shared interface, located via
  /// `neighbor_gid`/`neighbor_dim`/`neighbor_is_upper` exactly as recorded
  /// by `mesh::Mesh::connect()` (see `FaceBalance`) -- and folds
  /// `outgoing - incoming` into that patch's `leakage`, the same way
  /// `REFLECTIVE` faces already do at assembly time. Requires `table` to
  /// already be gathered (every patch present, see gather_patch_balances());
  /// purely local afterward, no further MPI. Leaves `incoming` unset for any
  /// `INTERNAL` face whose neighbor patch isn't present in `table`.
  static void resolve_internal_faces(std::vector<physics::PatchBalance>& table)
  {
    struct FaceKey {
      int64_t gid;
      size_t dim;
      bool is_upper;
      bool operator==(const FaceKey& other) const noexcept
      {
        return gid == other.gid && dim == other.dim &&
               is_upper == other.is_upper;
      }
    };
    struct FaceKeyHash {
      size_t operator()(const FaceKey& key) const noexcept
      {
        size_t h = std::hash<int64_t>()(key.gid);
        h ^= std::hash<size_t>()(key.dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>()(key.is_upper) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
      }
    };

    // Every INTERNAL face's own outgoing current, keyed by its own (gid,
    // dim, is_upper) -- built once, up front, so the resolution pass below
    // never mutates a PatchBalance while another entry still holds a
    // pointer into it.
    std::unordered_map<FaceKey, const torch::Tensor*, FaceKeyHash>
      outgoing_by_face;
    for (const auto& pb : table) {
      for (const auto& face : pb.faces) {
        if (face.type == physics::BoundaryType::INTERNAL) {
          outgoing_by_face.emplace(
            FaceKey {pb.gid, face.dim, face.is_upper}, &face.outgoing);
        }
      }
    }

    for (auto& pb : table) {
      for (auto& face : pb.faces) {
        if (face.type != physics::BoundaryType::INTERNAL ||
            !face.neighbor_gid.has_value() || !face.neighbor_dim.has_value() ||
            !face.neighbor_is_upper.has_value()) {
          continue;
        }
        auto it = outgoing_by_face.find(FaceKey {
          *face.neighbor_gid, *face.neighbor_dim, *face.neighbor_is_upper});
        if (it == outgoing_by_face.end()) {
          continue;
        }
        face.incoming = *it->second;
        pb.leakage += face.outgoing - *face.incoming;
      }
    }
  }

  // =================================================================
  // Private constructors
  TransportSolution(parallel::Communicator comm, typename Mesh::Ptr mesh,
    math::QuadratureSet::Ptr angular_qset, xs::Server::Ptr xs_server,
    physics::DGTransportAssemblerConfig config,
    std::optional<double> k_eff = std::nullopt,
    bool has_angular_dependence = true,
    std::optional<std::string> label = std::nullopt)
    : comm_(std::move(comm)), mesh_(std::move(mesh)),
      angular_qset_(std::move(angular_qset)), xs_server_(std::move(xs_server)),
      config_(std::move(config)), k_eff_(k_eff),
      has_angular_dependence_(has_angular_dependence),
      label_(label.has_value() ? Label::from_string(*label)
                               : Label::create_internal())
  {}

public:
  // =================================================================
  // Public methods
  /// @brief Build a TransportSolution and get the shared pointer to it.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new TransportSolution<BlockType>(std::forward<Args>(args)...));
  }

  /// @brief Add one of this rank's own local patches' field.
  /// @param gid Global ID of the mesh block this field belongs to.
  /// @param field The field for this patch (e.g. the solved angular flux).
  void add_local_patch(int64_t gid, linalg::State field)
  {
    local_fields_[gid] = std::move(field);
  }

  /// @brief Reduce every local field to its scalar flux (0th angular moment)
  /// via angular_qset_->integrate(). Returns a NEW TransportSolution holding
  /// the result -- this instance is untouched. k_eff/gid2rank/angular_qset/
  /// mesh carry over unchanged.
  /// @param eps TT-rounding tolerance applied after the contraction.
  /// @param max_rank TT-rounding max rank applied after the contraction.
  /// @throws ttnte::utils::runtime_error If no angular quadrature set is
  /// available (e.g. assemble() was never called on the originating driver).
  Ptr compute_scalar_flux(double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const
  {
    if (!angular_qset_) {
      throw utils::runtime_error(*this, error_context("compute_scalar_flux"),
        "No angular quadrature set is available. Was assemble() called on "
        "the originating driver?");
    }

    auto result = create(parallel::Communicator::world(), mesh_, angular_qset_,
      xs_server_, config_, k_eff_, /*has_angular_dependence=*/false);
    for (const auto& [gid, field] : local_fields_) {
      result->local_fields_[gid] =
        angular_qset_->integrate(field, eps, max_rank);
    }
    return result;
  }

  /// @brief Select a single energy group, narrowing every local field's
  /// energy axis (always the last core, regardless of whether angular
  /// dependence has been reduced) down to size 1. Returns a NEW
  /// TransportSolution holding the result -- this instance is untouched.
  /// k_eff/gid2rank/angular_qset/mesh/has_angular_dependence carry over
  /// unchanged. Works equally on the raw angular flux or an
  /// already-spatial-only (compute_scalar_flux()'d) solution -- selecting a
  /// group doesn't depend on angular reduction.
  /// @param group Index of the energy group to keep (0-based).
  /// @throws ttnte::utils::runtime_error If `group` is out of range for any
  /// of this rank's own local fields.
  Ptr select_group(int64_t group) const
  {
    auto result = create(parallel::Communicator::world(), mesh_, angular_qset_,
      xs_server_, config_, k_eff_, has_angular_dependence_);
    for (const auto& [gid, field] : local_fields_) {
      c10::SmallVector<int64_t, 6> m_modes =
        std::visit([](const auto& engine) { return engine.get_m_modes(); },
          field.get_variant());
      int64_t energy_dim = static_cast<int64_t>(m_modes.size()) - 1;
      int64_t num_groups = m_modes[energy_dim];

      if (group < 0 || group >= num_groups) {
        throw utils::runtime_error(*this, error_context("select_group"),
          "group " + std::to_string(group) + " is out of range for GID " +
            std::to_string(gid) + " (" + std::to_string(num_groups) +
            " energy groups)");
      }

      result->local_fields_[gid] = field.narrow(energy_dim, group, 1);
    }
    return result;
  }

  /// @brief Find this rank's own mesh block and the reference's mesh block
  /// for `gid`, plus the reference's local field -- the lookups shared by
  /// compute_errors() and error_norm().
  /// @throws ttnte::utils::runtime_error If no reference field, or no local
  /// mesh block on either side, is found for `gid`.
  std::tuple<typename Mesh::BlockTypePtr, typename Mesh::BlockTypePtr,
    const linalg::State*>
  find_error_inputs(int64_t gid, const TransportSolution& reference) const
  {
    auto ref_it = reference.local_fields_.find(gid);
    if (ref_it == reference.local_fields_.end()) {
      throw utils::runtime_error(*this, error_context("find_error_inputs"),
        "No reference field found for local GID " + std::to_string(gid) +
          " -- the reference TransportSolution must be distributed so that "
          "every GID this rank owns is also local on the reference");
    }

    typename Mesh::BlockTypePtr block;
    for (const auto& b : mesh_->get_blocks()) {
      if (b->get_gid() == gid) {
        block = b;
        break;
      }
    }
    if (!block) {
      throw utils::runtime_error(*this, error_context("find_error_inputs"),
        "No local mesh block found for GID " + std::to_string(gid));
    }

    typename Mesh::BlockTypePtr ref_block;
    for (const auto& b : reference.mesh_->get_blocks()) {
      if (b->get_gid() == gid) {
        ref_block = b;
        break;
      }
    }
    if (!ref_block) {
      throw utils::runtime_error(*this, error_context("find_error_inputs"),
        "No reference mesh block found for GID " + std::to_string(gid));
    }

    return {block, ref_block, &ref_it->second};
  }

  /// @brief Per-energy-group squared numerator (weighted sum of squared
  /// diff) and denominator (weighted sum of squared reference) for one
  /// patch -- shared by compute_errors() (which takes sqrt per patch, no
  /// MPI) and error_norm() (which sums across all local patches AND ranks
  /// before taking sqrt once, for the whole-solution error).
  ///
  /// Because the two sides may live on different DOF grids, they are never
  /// compared by subtracting DOF/control-point coefficients directly (only
  /// valid when both bases are identical). Instead both are evaluated at
  /// the finer side's own quadrature points first (see
  /// compute_error_weights_impl() for why), then subtracted, squared, and
  /// weighted -- with the energy axis left unreduced.
  /// @return {numerator, denominator}, both length-num_groups tensors,
  /// clamped to be non-negative (squares are mathematically non-negative;
  /// tiny negative values can appear from floating-point noise when the
  /// true value is ~0, e.g. comparing a solution against itself).
  std::pair<torch::Tensor, torch::Tensor> compute_error_terms(
    const typename Mesh::BlockTypePtr& block,
    const typename Mesh::BlockTypePtr& ref_block, const linalg::State& field,
    const linalg::State& ref_field) const
  {
    int64_t patch_ndim = block->get_ndim();
    int64_t num_angular_cores = has_angular_dependence_
                                  ? (angular_qset_->is_tensor_product() ? 2 : 1)
                                  : 0;

    // std::visit is needed ONLY to build the operators below -- it's the
    // one place that must reach into engine-specific internals (cores,
    // m_modes) to assemble new TT cores. Everything after this uses only
    // the already format-generic Operator/State APIs (mv, arithmetic,
    // to_dense()), so it lives outside the dispatch.
    auto [eval_op, ref_eval_op, weight_op, num_groups] = std::visit(
      [&](const auto& engine) -> std::tuple<linalg::Operator, linalg::Operator,
                                linalg::Operator, int64_t> {
        using EngineType = std::decay_t<decltype(engine)>;

        if constexpr (std::is_same_v<EngineType, linalg::TTEngine>) {
          const auto& m_modes = engine.get_m_modes();
          int64_t num_groups = m_modes.back();
          auto device = engine.get_device();
          auto dtype = engine.get_dtype();

          auto [basis, mapping, ref_basis] =
            compute_error_weights<linalg::FormatType::TENSOR_TRAIN>(
              block, ref_block);

          // Shared identity(angle)/identity(energy) blocks, reused for both
          // this field's and the reference's evaluation operator -- angular
          // DOF count and number of groups are assumed identical between
          // the two (only the spatial discretization may differ).
          linalg::TTEngine::Tensors id_angle_cores;
          linalg::TTEngine::Tensors weight_cores;

          if (num_angular_cores > 0) {
            c10::SmallVector<int64_t, 6> angle_modes(
              m_modes.begin(), m_modes.begin() + num_angular_cores);
            auto id_angle =
              linalg::TTEngine::ones(angle_modes, device, dtype).diagonalize();
            for (const auto& c : id_angle.get_cores()) {
              id_angle_cores.push_back(c);
            }

            if (angular_qset_->is_tensor_product()) {
              auto product_qset =
                std::static_pointer_cast<math::ProductQuadrature>(
                  angular_qset_);
              for (const auto& w : product_qset->get_factored_weights()) {
                weight_cores.push_back(w.reshape({1, 1, -1, 1}));
              }
            } else {
              weight_cores.push_back(
                angular_qset_->get_weights().reshape({1, 1, -1, 1}));
            }
          }

          // mapping's cores are shaped as a value at each quadrature point
          // (m = num_quad_points, n = 1); swap m/n so they act as a
          // reduction operator, matching the angular weight convention.
          for (int64_t d = 0; d < patch_ndim; d++) {
            weight_cores.push_back(
              mapping[d].permute({0, 2, 1, 3}).contiguous());
          }

          c10::SmallVector<int64_t, 6> energy_modes {num_groups};
          auto id_energy =
            linalg::TTEngine::ones(energy_modes, device, dtype).diagonalize();
          weight_cores.push_back(id_energy[0]);

          auto build_eval_op =
            [&](const linalg::TTEngine& spatial_basis) -> linalg::Operator {
            linalg::TTEngine::Tensors cores = id_angle_cores;
            for (const auto& c : spatial_basis.get_cores()) {
              cores.push_back(c);
            }
            cores.push_back(id_energy[0]);
            linalg::TTEngine eval_engine(cores);
            return linalg::Operator(eval_engine);
          };

          linalg::Operator eval_op = build_eval_op(basis);
          linalg::Operator ref_eval_op = build_eval_op(ref_basis);
          linalg::Operator weight_op =
            linalg::Operator(linalg::TTEngine(weight_cores));

          return {eval_op, ref_eval_op, weight_op, num_groups};
        } else {
          throw utils::runtime_error(*this, error_context("compute_errors"),
            "This State format is not supported yet");
        }
      },
      field.get_variant());

    // Evaluate both sides at the shared (finer side's) quadrature points
    // before comparing -- never subtract DOF/control-point coefficients
    // directly, which is only valid when both bases are identical. All of
    // this is generic Operator/State arithmetic -- mv()/State's own
    // operators already dispatch on format internally, so none of this
    // needs to be inside the visit above.
    linalg::State at_quad_field = linalg::mv(eval_op, field);
    linalg::State at_quad_ref = linalg::mv(ref_eval_op, ref_field);
    linalg::State diff = at_quad_field - at_quad_ref;
    linalg::State sq_diff = diff * diff;
    linalg::State sq_ref = at_quad_ref * at_quad_ref;

    // weight_op already reduces every angle/space mode down to size 1 via
    // the mv() contraction itself (angular quadrature weight + the
    // Jacobian-weighted spatial mapping), leaving only the energy axis -- so
    // to_dense() + reshape is all that's needed, no further summation.
    torch::Tensor numerator =
      linalg::mv(weight_op, sq_diff).to_dense().reshape({num_groups});
    torch::Tensor denominator =
      linalg::mv(weight_op, sq_ref).to_dense().reshape({num_groups});

    numerator.clamp_min_(0.0);
    denominator.clamp_min_(0.0);

    return {numerator, denominator};
  }

  /// @brief Per-patch, per-energy-group relative L2 error against a
  /// reference TransportSolution, for this rank's own local patches only --
  /// no MPI. Assumes shared geometry: the reference's mesh must have a
  /// block for every GID this rank owns, on the SAME parametric domain, but
  /// the reference's NURBS discretization (knot spans, polynomial degree)
  /// may differ, e.g. a higher-fidelity verification solve.
  /// @param reference The reference solution (e.g. a finer/higher-degree
  /// solve of the same geometry). Must hold a local field for every GID this
  /// rank owns, and must match this solution's has_angular_dependence_.
  /// @return GID -> length-num_groups tensor of
  /// sqrt(sum(diff^2) / sum(reference^2)) per group, properly angle- and
  /// volume-weighted -- only for this rank's own local GIDs.
  /// @throws ttnte::utils::runtime_error If has_angular_dependence_ doesn't
  /// match `reference`'s, or no reference/mesh block is found for one of
  /// this rank's own local GIDs.
  std::unordered_map<int64_t, torch::Tensor> compute_errors(
    const TransportSolution& reference) const
  {
    if (has_angular_dependence_ != reference.has_angular_dependence_) {
      throw utils::runtime_error(*this, error_context("compute_errors"),
        "This solution and the reference must both hold the same kind of "
        "field (both angular flux, or both already reduced via "
        "compute_scalar_flux())");
    }

    std::unordered_map<int64_t, torch::Tensor> result;
    for (const auto& [gid, field] : local_fields_) {
      auto [block, ref_block, ref_field] = find_error_inputs(gid, reference);
      auto [numerator, denominator] =
        compute_error_terms(block, ref_block, field, *ref_field);
      result[gid] = (numerator / denominator).sqrt();
    }
    return result;
  }

  /// @brief Relative L2 error per energy group against a reference
  /// TransportSolution, aggregated over EVERY local patch on EVERY rank --
  /// always collective (every rank must call this; matches
  /// solve_eigenvalue()'s own convergence-check pattern of a single small
  /// iallreduce(SUM), not a gather of the full per-patch breakdown). Sums
  /// the numerator and denominator across all local patches first (energy
  /// axis left unreduced), combines across ranks with one iallreduce(SUM),
  /// and takes sqrt of the combined ratio last -- the properly weighted
  /// whole-solution error per group, not an average of per-patch ratios.
  /// @param reference The reference solution (e.g. a finer/higher-degree
  /// solve of the same geometry). Must hold a local field for every GID
  /// every rank owns, and must match this solution's
  /// has_angular_dependence_.
  /// @return A length-num_groups tensor of sqrt(sum(diff^2) /
  /// sum(reference^2)) per group, properly weighted and reduced over every
  /// patch on every rank -- identical on every rank.
  /// @throws ttnte::utils::runtime_error If has_angular_dependence_ doesn't
  /// match `reference`'s, or no reference/mesh block is found for one of
  /// this rank's own local GIDs.
  torch::Tensor error_norm(const TransportSolution& reference) const
  {
    if (has_angular_dependence_ != reference.has_angular_dependence_) {
      throw utils::runtime_error(*this, error_context("error_norm"),
        "This solution and the reference must both hold the same kind of "
        "field (both angular flux, or both already reduced via "
        "compute_scalar_flux())");
    }

    torch::Tensor local_numerator, local_denominator;
    int64_t local_num_groups = 0;

    for (const auto& [gid, field] : local_fields_) {
      auto [block, ref_block, ref_field] = find_error_inputs(gid, reference);
      auto [numerator, denominator] =
        compute_error_terms(block, ref_block, field, *ref_field);

      if (local_num_groups == 0) {
        local_numerator = numerator.clone();
        local_denominator = denominator.clone();
        local_num_groups = numerator.size(0);
      } else {
        local_numerator += numerator;
        local_denominator += denominator;
      }
    }

    int64_t num_groups = local_num_groups;
    if (comm_.size() > 1) {
      int64_t global_num_groups = 0;
      comm_
        .iallreduce(
          &local_num_groups, &global_num_groups, 1, parallel::MPIOp::MAX)
        .wait();
      num_groups = global_num_groups;
    }

    if (!local_numerator.defined()) {
      auto options = torch::TensorOptions().dtype(torch::kFloat64);
      local_numerator = torch::zeros({num_groups}, options);
      local_denominator = torch::zeros({num_groups}, options);
    }

    // Combine [numerator | denominator] into one buffer for a single
    // iallreduce, generalizing the same combined-reduction pattern
    // TransportDriver::solve_eigenvalue()'s own convergence check uses, now
    // per-group instead of scalar. The Communicator's iallreduce only
    // supports double buffers, but local_numerator/local_denominator carry
    // whatever dtype the solve itself used (e.g. float32) -- cast for the
    // wire, then cast the reduced result back.
    torch::Tensor local_sums = torch::cat({local_numerator, local_denominator});
    torch::ScalarType result_dtype = local_sums.scalar_type();
    torch::Tensor global_sums = local_sums;
    if (comm_.size() > 1) {
      torch::Tensor local_sums_f64 = local_sums.to(torch::kFloat64);
      torch::Tensor global_sums_f64 = torch::empty_like(local_sums_f64);
      comm_
        .iallreduce(local_sums_f64.data_ptr<double>(),
          global_sums_f64.data_ptr<double>(),
          static_cast<int>(local_sums_f64.numel()), parallel::MPIOp::SUM)
        .wait();
      global_sums = global_sums_f64.to(result_dtype);
    }

    torch::Tensor numerator =
      global_sums.narrow(0, 0, num_groups).clamp_min(0.0);
    torch::Tensor denominator =
      global_sums.narrow(0, num_groups, num_groups).clamp_min(0.0);

    return (numerator / denominator).sqrt();
  }

  /// @brief Volume-averaged field on a regular Cartesian grid, via
  /// composite trapezoidal integration over `n` sub-points per cell.
  /// Requires this solution to be spatial-only (the result of
  /// compute_scalar_flux()). Assumes axis-aligned boundaries -- composite
  /// trapezoidal integration treats each grid cell as a plain box, matching
  /// the same caveat the legacy ttnte/iga/mesh.py implementation this
  /// generalizes documents.
  ///
  /// Always collective (every rank must call this): every rank redundantly
  /// builds the identical regular grid -- the physical domain's bounding
  /// box is auto-computed via one small iallreduce(MIN)/iallreduce(MAX)
  /// over each rank's own local patches' bboxes (no geometry ever crosses
  /// ranks, just phys_dim-sized scalars). For each of THIS rank's own local
  /// patches, candidate grid points inside that patch's (padded) bbox are
  /// inverse-mapped and evaluated locally (bbox padding absorbs
  /// floating-point boundary mismatches between neighboring patches,
  /// matching the legacy tolerance). Because two neighboring patches'
  /// padded bboxes can overlap even when the patches themselves don't,
  /// grid-point ownership is resolved via iallreduce(MIN) on the
  /// Newton-Raphson residual (every rank learns the globally-best residual
  /// per point), then iallreduce(SUM) on a winner-only contribution (every
  /// rank ends up with the same, ownership-resolved field value at every
  /// grid point; ties are vanishingly unlikely with independent
  /// floating-point residuals from geometrically distinct patches and
  /// aren't specially handled).
  /// @param shape Number of cells along each axis; its length fixes the
  /// dimensionality (1, 2, or 3).
  /// @param n Number of sub-points per cell, per axis, for trapezoidal
  /// integration (must be > 1 in every axis); same length as `shape`.
  /// @param max_iter Max Newton-Raphson iterations for Patch::inverse_map().
  /// Every candidate point that's inside a patch's bounding box but outside
  /// the patch's actual (possibly curved) boundary can never converge, so
  /// it burns every iteration regardless of this cap -- keep this low
  /// (genuine points converge in well under 10 iterations once seeded via
  /// seed_resolution) rather than raising it to chase spurious non-converged
  /// points.
  /// @param tol Convergence tolerance for Patch::inverse_map().
  /// @param seed_resolution Points per parametric axis for Patch::
  /// inverse_map()'s coarse-grid seeding (see DEFAULT_INVERSE_MAP_SEED_
  /// RESOLUTION). Raise this if the "not covered by any patch" error below
  /// fires spuriously -- e.g. a grid point genuinely on the mesh but near a
  /// coordinate singularity or a multi-patch corner can need a finer seed
  /// than the default to keep Newton-Raphson in the right basin.
  /// @return A tensor of shape (*shape, num_groups): the volume-averaged
  /// field per cell per group, identical on every rank.
  /// @throws ttnte::utils::runtime_error If this solution still carries
  /// angular dependence, if `shape`/`n` sizes mismatch or any `n[d] <= 1`,
  /// or if some grid point is not covered by any patch on any rank.
  torch::Tensor regular_mesh_average(c10::SmallVector<int64_t, 3> shape,
    c10::SmallVector<int64_t, 3> n, int64_t max_iter = 10, double tol = 1e-8,
    int64_t seed_resolution = cad::DEFAULT_INVERSE_MAP_SEED_RESOLUTION) const
  {
    if (has_angular_dependence_) {
      throw utils::runtime_error(*this, error_context("regular_mesh_average"),
        "This solution must be spatial-only -- call compute_scalar_flux() "
        "first");
    }
    if (shape.size() != n.size()) {
      throw utils::runtime_error(*this, error_context("regular_mesh_average"),
        "`shape` and `n` must have the same length");
    }
    int64_t phys_dim = static_cast<int64_t>(shape.size());
    for (int64_t d = 0; d < phys_dim; d++) {
      if (n[d] <= 1) {
        throw utils::runtime_error(*this, error_context("regular_mesh_average"),
          "Every entry of `n` must be > 1");
      }
    }

    constexpr double bbox_padding = 5e-5;
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    // -- Global bounding box: local min/max over this rank's own patches,
    // combined across ranks via one iallreduce(MIN)/iallreduce(MAX) (tiny,
    // phys_dim-sized -- no geometry crosses ranks).
    torch::Tensor local_min =
      torch::full({phys_dim}, std::numeric_limits<double>::infinity(), options);
    torch::Tensor local_max = torch::full(
      {phys_dim}, -std::numeric_limits<double>::infinity(), options);
    for (const auto& block : mesh_->get_blocks()) {
      torch::Tensor bbox = block->get_bbox().to(options);
      local_min = torch::minimum(local_min, bbox[0]);
      local_max = torch::maximum(local_max, bbox[1]);
    }

    torch::Tensor global_min = local_min.clone();
    torch::Tensor global_max = local_max.clone();
    if (comm_.size() > 1) {
      comm_
        .iallreduce(local_min.data_ptr<double>(), global_min.data_ptr<double>(),
          phys_dim, parallel::MPIOp::MIN)
        .wait();
      comm_
        .iallreduce(local_max.data_ptr<double>(), global_max.data_ptr<double>(),
          phys_dim, parallel::MPIOp::MAX)
        .wait();
    }

    // -- Build the regular grid: per-axis (shape[d], n[d]) sub-point arrays
    // (one independent linspace per cell, not one grid-wide linspace), then
    // their ND tensor product gives every physical sample point.
    std::vector<torch::Tensor> axis_flat;
    axis_flat.reserve(phys_dim);
    for (int64_t d = 0; d < phys_dim; d++) {
      torch::Tensor edges = torch::linspace(global_min[d].item<double>(),
        global_max[d].item<double>(), shape[d] + 1, options);
      torch::Tensor left = edges.narrow(0, 0, shape[d]);
      torch::Tensor width = edges.diff();
      torch::Tensor t = torch::linspace(0.0, 1.0, n[d], options);
      torch::Tensor axis_points =
        left.unsqueeze(1) + width.unsqueeze(1) * t.unsqueeze(0);
      axis_flat.push_back(axis_points.flatten());
    }

    std::vector<torch::Tensor> grids = torch::meshgrid(axis_flat, "ij");
    torch::Tensor points =
      torch::stack(grids, -1); // (L0, ..., L_{d-1}, phys_dim)
    int64_t total_points = points.numel() / phys_dim;
    torch::Tensor flat_points = points.reshape({total_points, phys_dim});

    // -- Determine num_groups (even if this rank owns zero local patches).
    int64_t local_num_groups = 0;
    if (!local_fields_.empty()) {
      const auto& [any_gid, any_field] = *local_fields_.begin();
      typename Mesh::BlockTypePtr any_block;
      for (const auto& b : mesh_->get_blocks()) {
        if (b->get_gid() == any_gid) {
          any_block = b;
          break;
        }
      }
      torch::Tensor dense = any_field.to_dense();
      local_num_groups = dense.size(any_block->get_ndim());
    }
    int64_t num_groups = local_num_groups;
    if (comm_.size() > 1) {
      int64_t global_num_groups = 0;
      comm_
        .iallreduce(
          &local_num_groups, &global_num_groups, 1, parallel::MPIOp::MAX)
        .wait();
      num_groups = global_num_groups;
    }

    // -- For each of this rank's own local patches: bbox-filter candidate
    // grid points, inverse-map, and evaluate locally.
    torch::Tensor local_residual = torch::full(
      {total_points}, std::numeric_limits<double>::infinity(), options);
    torch::Tensor local_values =
      torch::zeros({total_points, num_groups}, options);

    for (const auto& block : mesh_->get_blocks()) {
      torch::Tensor padded_bbox = block->get_bbox(bbox_padding).to(options);
      torch::Tensor bbox_min = padded_bbox[0];
      torch::Tensor bbox_max = padded_bbox[1];

      torch::Tensor in_bbox =
        torch::logical_and((flat_points >= bbox_min.unsqueeze(0)).all(-1),
          (flat_points <= bbox_max.unsqueeze(0)).all(-1));
      torch::Tensor candidate_idx = in_bbox.nonzero().squeeze(-1);
      if (candidate_idx.numel() == 0) {
        continue;
      }

      torch::Tensor candidate_points =
        flat_points.index_select(0, candidate_idx);
      auto imr = block->inverse_map(
        candidate_points, max_iter, tol, std::nullopt, seed_resolution);

      torch::Tensor converged_local_idx = imr.converged.nonzero().squeeze(-1);
      if (converged_local_idx.numel() == 0) {
        continue;
      }

      torch::Tensor converged_global_idx =
        candidate_idx.index_select(0, converged_local_idx);
      torch::Tensor converged_coords =
        imr.coords.index_select(0, converged_local_idx);
      torch::Tensor converged_residual =
        imr.residual.index_select(0, converged_local_idx);

      torch::Tensor values = evaluate_field_at_points(
        block, local_fields_.at(block->get_gid()), converged_coords);

      // Only accept this block's contribution where it strictly improves on
      // whatever this rank has already recorded for the same point. Points
      // exactly on a shared patch boundary (e.g. two ruled patches meeting at
      // a seam) can be valid, converged candidates for more than one LOCAL
      // block -- without this check, whichever block happens to be last in
      // mesh_->get_blocks() iteration order would silently clobber an
      // earlier, equally-valid entry via index_copy_. Using strict `<`
      // (not `<=`) means the first block to claim a point keeps it on an
      // exact tie, which is deterministic and, critically, mirrors the
      // cross-rank tie-break below so a point's winner doesn't depend on how
      // many ranks the mesh happens to be split across.
      torch::Tensor current_residual =
        local_residual.index_select(0, converged_global_idx);
      torch::Tensor improves = converged_residual < current_residual;
      torch::Tensor improve_local_idx = improves.nonzero().squeeze(-1);
      if (improve_local_idx.numel() == 0) {
        continue;
      }
      torch::Tensor improve_global_idx =
        converged_global_idx.index_select(0, improve_local_idx);

      local_residual.index_copy_(0, improve_global_idx,
        converged_residual.index_select(0, improve_local_idx));
      local_values.index_copy_(
        0, improve_global_idx, values.index_select(0, improve_local_idx));
    }

    // -- Resolve ownership across ranks: MIN residual, then a deterministic
    // rank tie-break so exactly one rank contributes per point. A plain
    // `local_residual <= global_residual` comparison (the previous approach)
    // lets EVERY rank that exactly ties the global minimum count as a
    // winner -- and exact ties are the common case right at a shared patch
    // boundary, where two neighboring patches' inverse_map both converge to
    // a near-zero (sometimes bit-identical) residual. When those patches
    // happen to live on different ranks, the winner-only SUM below would
    // then add both patches' field values together instead of picking one,
    // producing a sharp, rank-distribution-dependent artifact exactly at
    // patch interfaces. Breaking ties by lowest rank makes the winner -- and
    // therefore the averaged output -- independent of how the mesh happens
    // to be partitioned.
    torch::Tensor global_residual = local_residual;
    if (comm_.size() > 1) {
      global_residual = torch::empty_like(local_residual);
      comm_
        .iallreduce(local_residual.data_ptr<double>(),
          global_residual.data_ptr<double>(), total_points,
          parallel::MPIOp::MIN)
        .wait();
    }

    if (torch::isinf(global_residual).any().item<bool>()) {
      throw utils::runtime_error(*this, error_context("regular_mesh_average"),
        "At least one grid point is not covered by any patch on any rank");
    }

    torch::Tensor is_winner;
    if (comm_.size() > 1) {
      auto rank_options = torch::TensorOptions().dtype(torch::kInt32);
      torch::Tensor candidate_rank =
        torch::where(local_residual <= global_residual,
          torch::full({total_points}, comm_.rank(), rank_options),
          torch::full({total_points}, comm_.size(), rank_options));
      torch::Tensor winning_rank = torch::empty_like(candidate_rank);
      comm_
        .iallreduce(candidate_rank.data_ptr<int32_t>(),
          winning_rank.data_ptr<int32_t>(), total_points, parallel::MPIOp::MIN)
        .wait();
      is_winner =
        (winning_rank == comm_.rank()).unsqueeze(-1).to(local_values.dtype());
    } else {
      is_winner = (local_residual <= global_residual)
                    .unsqueeze(-1)
                    .to(local_values.dtype());
    }
    torch::Tensor contribution = local_values * is_winner;

    torch::Tensor global_values = contribution;
    if (comm_.size() > 1) {
      global_values = torch::empty_like(contribution);
      comm_
        .iallreduce(contribution.data_ptr<double>(),
          global_values.data_ptr<double>(), total_points * num_groups,
          parallel::MPIOp::SUM)
        .wait();
    }

    // -- Composite trapezoidal integration per cell: reshape the flat
    // per-point values back into (shape_0, n_0, ..., shape_{d-1}, n_{d-1},
    // num_groups) and weight/sum only the per-axis sub-point dimensions.
    c10::SmallVector<int64_t, 8> full_shape;
    full_shape.reserve(2 * phys_dim + 1);
    for (int64_t d = 0; d < phys_dim; d++) {
      full_shape.push_back(shape[d]);
      full_shape.push_back(n[d]);
    }
    full_shape.push_back(num_groups);
    torch::Tensor reshaped_values = global_values.reshape(full_shape);

    // Composite trapezoidal weight per axis: 1/2 at the two ends, 1
    // elsewhere; the combined ND weight is the outer product across axes.
    torch::Tensor weight;
    double normalization = 1.0;
    for (int64_t d = 0; d < phys_dim; d++) {
      torch::Tensor w = torch::ones({n[d]}, options);
      w[0] = 0.5;
      w[n[d] - 1] = 0.5;
      normalization *= static_cast<double>(n[d] - 1);

      c10::SmallVector<int64_t, 8> w_shape(2 * phys_dim + 1, 1);
      w_shape[2 * d + 1] = n[d];
      torch::Tensor w_broadcast = w.reshape(w_shape);
      weight = weight.defined() ? weight * w_broadcast : w_broadcast;
    }

    torch::Tensor weighted = reshaped_values * weight;

    std::vector<int64_t> sum_dims;
    sum_dims.reserve(phys_dim);
    for (int64_t d = 0; d < phys_dim; d++) {
      sum_dims.push_back(2 * d + 1);
    }
    // Summing the sub-point axes leaves the cell axes and num_groups in
    // their original relative order.
    torch::Tensor cell_sums = weighted.sum(sum_dims);

    return cell_sums / normalization;
  }

  /// @brief Compute every one of this rank's own local patches' own
  /// particle-balance diagnostics (source/absorption/scatter/leakage per
  /// group, per face) from this solution's own field -- purely local, no
  /// MPI (mirrors compute_errors()'s own "no MPI" convention). Cross-patch
  /// resolution of INTERNAL faces (the only place particle balance needs
  /// data shared between patches) happens in patch_balance_table()/
  /// global_balance(), not here.
  /// @param assemblers GID -> assembler, for as many of this rank's own
  /// local patches as the caller already has on hand -- e.g.
  /// TransportDriver::get_assemblers(), called with clear_assemblers=false
  /// so the assemblers survive the solve. Optional: any local GID missing
  /// from this map (including every GID, if the map is left empty/omitted)
  /// gets a fresh assembler built on demand via make_assembler(), so this
  /// solution never strictly needs the full TransportDriver -- just the
  /// xs_server_/config_ it was already given at construction. Building a
  /// fresh assembler re-runs that patch's full assemble(), so passing
  /// already-built assemblers is strictly cheaper when they're available.
  /// @param eps TT-rounding tolerance for the balance/leakage functionals'
  /// own construction -- independent of the solve's own tolerance, so
  /// eps=0 gives an exact/benchmarking-grade result.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return GID -> PatchBalance, for this rank's own local patches only.
  std::unordered_map<int64_t, physics::PatchBalance> compute_patch_balances(
    const std::unordered_map<int64_t, AssemblerPtr>& assemblers = {},
    double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const
  {
    std::unordered_map<int64_t, physics::PatchBalance> result;
    for (const auto& [gid, field] : local_fields_) {
      auto it = assemblers.find(gid);
      AssemblerPtr assembler =
        it != assemblers.end() ? it->second : make_assembler_for_gid(gid);
      result.emplace(gid, assembler->compute_balance(field, eps, max_rank));
    }
    return result;
  }

  /// @brief Every patch's own particle-balance diagnostics -- fixed source,
  /// absorption, scatter-in/out, fission, leakage, and a per-face breakdown
  /// -- each computed purely from that patch's own solution vector, gathered
  /// across every rank and sorted by GID, with every `INTERNAL` face then
  /// resolved against its neighbor's own `outgoing` (see
  /// resolve_internal_faces()) so each patch's own balance closes on its
  /// own, `INTERNAL` faces included. Collective (every rank must call this);
  /// the result is identical on every rank (see gather_patch_balances()).
  /// @param assemblers GID -> assembler, optional (see
  /// compute_patch_balances()).
  /// @param eps TT-rounding tolerance, independent of the solve's own.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return Every patch's PatchBalance, sorted by GID.
  physics::PatchBalanceTable patch_balance_table(
    const std::unordered_map<int64_t, AssemblerPtr>& assemblers = {},
    double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const
  {
    physics::PatchBalanceTable table;
    table.patches =
      gather_patch_balances(compute_patch_balances(assemblers, eps, max_rank));
    resolve_internal_faces(table.patches);
    return table;
  }

  /// @brief Particle-balance diagnostics summed over the whole problem.
  /// Collective; identical on every rank. `leakage` sums every patch's own
  /// (resolved -- see resolve_internal_faces()) `leakage` directly: each
  /// `INTERNAL` interface's two sides are resolved from the same pair of
  /// `outgoing` tensors with opposite sign, so they cancel out exactly in
  /// this sum and no separate exclusion is needed for the true whole-system
  /// leakage. `dd_residual` resolves every INTERNAL interface exactly once
  /// (via `gid < neighbor_gid`) by comparing the two patches' independently,
  /// locally computed outgoing currents through their shared face -- ~0 for
  /// a lossless, fully converged distributed solve, and the direct "is the
  /// distributed method losing particles" diagnostic.
  /// @param assemblers GID -> assembler, optional (see
  /// compute_patch_balances()).
  /// @param eps TT-rounding tolerance, independent of the solve's own.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return The whole-problem GlobalBalance.
  physics::GlobalBalance global_balance(
    const std::unordered_map<int64_t, AssemblerPtr>& assemblers = {},
    double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const
  {
    std::vector<physics::PatchBalance> table =
      gather_patch_balances(compute_patch_balances(assemblers, eps, max_rank));
    resolve_internal_faces(table);

    int64_t num_groups = table.empty() ? 0 : table[0].fixed_source.numel();
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    physics::GlobalBalance result;
    result.fixed_source = torch::zeros({num_groups}, options);
    result.fission_source = torch::zeros({num_groups}, options);
    result.absorption = torch::zeros({num_groups}, options);
    result.scatter_in = torch::zeros({num_groups}, options);
    result.scatter_out = torch::zeros({num_groups}, options);
    result.leakage = torch::zeros({num_groups}, options);
    result.dd_residual = torch::zeros({num_groups}, options);

    for (const auto& pb : table) {
      result.fixed_source += pb.fixed_source;
      result.fission_source += pb.fission_source;
      result.absorption += pb.absorption;
      result.scatter_in += pb.scatter_in;
      result.scatter_out += pb.scatter_out;
      result.leakage += pb.leakage;

      for (const auto& face : pb.faces) {
        if (face.type != physics::BoundaryType::INTERNAL ||
            !face.neighbor_gid.has_value() || !face.incoming.has_value()) {
          continue;
        }
        // Count each interface exactly once, from its lower-GID side.
        if (pb.gid >= *face.neighbor_gid) {
          continue;
        }
        result.dd_residual += (face.outgoing - *face.incoming).abs();
      }
    }

    return result;
  }

  // =================================================================
  // Public getters / setters
  /// @return The label of the solution.
  const Label& get_label() const noexcept { return label_; }
  /// @return The converged k-eigenvalue, if set.
  std::optional<double> get_k_eff() const noexcept { return k_eff_; }
  /// @return GID -> owning rank, for every GID in the whole mesh. Delegates
  /// to mesh_'s own gid2rank_ -- see Mesh::get_gid2rank().
  const std::unordered_map<int64_t, int>& get_gid2rank() const noexcept
  {
    return mesh_->get_gid2rank();
  }

  /// @brief Get this rank's own local field for a GID (no MPI). Call
  /// .to_dense() on the result for a dense tensor.
  /// @param gid Global ID of the mesh block.
  /// @throws ttnte::utils::runtime_error If GID is not local to this rank.
  const linalg::State& get_local_field(int64_t gid) const
  {
    auto it = local_fields_.find(gid);
    if (it == local_fields_.end()) {
      throw utils::runtime_error(*this, error_context("get_local_field"),
        "GID " + std::to_string(gid) + " is not local to this rank");
    }
    return it->second;
  }
};

} // namespace ttnte::driver
