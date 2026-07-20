// Native AMEn sweep driver, used by amen_solve_native().
//
// Structurally mirrors torchTT's amen_solve
// (external/torchTT/torchtt/cpp/amen_solve.h) -- same two-half-sweep AMEn
// algorithm (Dolgov & Savostyanov 2013/2014; TT-Toolbox `amen_solve2.m`,
// Oseledets et al.) -- but built on ttnte's own primitives instead of
// torchTT's tensordot chains / AMENsolveMV / gmres.h:
//   - amen::FoldedLocalOperator instead of `local_product` / dense-B
//     tensordot assembly (folded GEMMs, cached across GMRES iterations).
//   - amen::gmres_solve instead of AMENsolveMV + gmres<double> (runs in the
//     input's own dtype; dispatches to the CPU incremental-Givens strategy
//     or the GPU fixed-budget-Arnoldi-plus-lstsq strategy based on device).
//   - amen::qless_orthogonalize instead of at::linalg_qr for pure
//     orthogonalization steps (Q-less, blocked TSQR).
//   - amen::Rank1Preconditioner as an optional global preconditioning wrap
//     around the sweep (Roehrig-Zoellner et al. 2025, Sec. 3.4.1).
//   - amen::fixed_rank_basis (below) for AMEnEnrichmentMode::ALS_FIXED_RANK
//     (pitts' "AMEn+ALS", Algorithm 5 of the same paper): a fixed-rank
//     residual/enrichment basis, updated via QR only (no SVD), instead of
//     the baseline's adaptive SVD-truncated-to-kickrank(+kick2) update.
// Independent reimplementation against ttnte's own tensor types; no code
// copied from torchTT (MIT, ion-g-ion) or pitts (BSD-3-Clause, DLR).
#include "ttnte/linalg/amen/amen_sweep.hpp"
#include "ttnte/linalg/amen/gmres_local.hpp"
#include "ttnte/linalg/amen/local_operator.hpp"
#include "ttnte/linalg/amen/local_preconditioner.hpp"
#include "ttnte/linalg/amen/phi_recursions.hpp"
#include "ttnte/linalg/amen/qless_tsqr.hpp"
#include "ttnte/linalg/amen/rank1_preconditioner.hpp"
#include "ttnte/utils/exception.hpp"
#include <cmath>
#include <iostream>

using namespace torch::indexing;

namespace ttnte::linalg::amen {

namespace {

/// @brief Orthogonalizes `M` via Q-less blocked TSQR when `use_qless` is
/// true (the default), or a plain dense `at::linalg_qr` otherwise --
/// `AMEnNativeOptions::use_qless_tsqr`'s only effect. Single change point
/// shared by every orthogonalization call in this file instead of an
/// inline branch at each one.
std::pair<torch::Tensor, torch::Tensor> orthogonalize_maybe_qless(
  const torch::Tensor& M, int64_t block_size, bool use_qless)
{
  if (use_qless) {
    return qless_orthogonalize(M, block_size);
  }
  auto [q, r] = torch::linalg_qr(M, "reduced");
  return {q, r};
}

/// @brief Orthonormal basis of exactly `target_rank` columns spanning (an
/// approximation of) `candidate`'s column space: pads with random columns if
/// `candidate` is narrower than `target_rank`, then keeps the first
/// `target_rank` columns of an `orthogonalize_maybe_qless` orthogonalization.
/// No SVD.
torch::Tensor fixed_rank_basis(const torch::Tensor& candidate,
  int64_t target_rank, int tsqr_block_size, bool use_qless_tsqr,
  const torch::TensorOptions& options)
{
  const int64_t rows = candidate.size(0);
  const int64_t rank = std::min(target_rank, rows);
  torch::Tensor padded = candidate;
  if (padded.size(1) < rank) {
    padded = torch::cat(
      {padded, torch::randn({rows, rank - padded.size(1)}, options)}, 1);
  }
  auto [Q, R] = orthogonalize_maybe_qless(padded, tsqr_block_size, use_qless_tsqr);
  return Q.index({Slice(), Slice(0, rank)});
}

} // namespace

std::vector<torch::Tensor> amen_sweep(std::vector<torch::Tensor> A_cores,
  std::vector<torch::Tensor> b_cores, std::vector<torch::Tensor> x_cores,
  const std::vector<int64_t>& N, int nswp, double eps, int64_t max_rank,
  int64_t max_full, int64_t kickrank, int64_t kick2, int local_iterations,
  int resets, bool verbose, AMEnPreconditioner preconditioner,
  int tsqr_block_size, bool use_qless_tsqr, AMEnEnrichmentMode mode,
  int64_t als_residual_rank, bool use_local_forcing,
  double gmres_forcing_ceiling, bool use_gpu_batched_gmres)
{
  if (preconditioner == AMEnPreconditioner::RANK1) {
    throw utils::runtime_error("ttnte::linalg::amen::amen_sweep",
      "`AMEnPreconditioner::RANK1` must be handled by the caller "
      "(amen_solve_dispatch) before reaching amen_sweep -- only NONE, "
      "LOCAL_C_PREC, and LOCAL_R_PREC are valid here");
  }
  // Matches the vendored torchTT baseline (amen_solve.h), which wraps its
  // whole sweep the same way -- every op below (SVD, GMRES, tensordot chains)
  // is otherwise unnecessarily tracked if called under a grad-enabled context.
  torch::NoGradGuard no_grad;

  const bool fixed_rank = (mode == AMEnEnrichmentMode::ALS_FIXED_RANK);
  // Requesting zero enrichment (e.g. AMEnSolver freezing the rank) collapses
  // rz[1..d-1] to 0 below. Building/orthogonalizing genuinely 0-sized
  // z_cores crashes (ambiguous `.reshape({0, -1})`), and nothing downstream
  // needs them -- every read of z_cores/Phiz/Phiz_b is already gated by
  // `!last`, so gating those same sites on `!enrichment_disabled` too (see
  // below) makes skipping their construction entirely safe: this becomes a
  // pure ALS sweep (local solve + truncation only, rank can shrink via the
  // usual eps/max_rank truncation loop but never grow).
  const bool enrichment_disabled =
    fixed_rank ? (als_residual_rank == 0) : (kickrank == 0 && kick2 == 0);

  auto options = A_cores[0].options();
  const int64_t d = static_cast<int64_t>(N.size());

  std::vector<int64_t> rx(d + 1);
  rx[0] = 1;
  for (int64_t i = 0; i < d; ++i) {
    rx[i + 1] = x_cores[i].size(2);
  }

  std::vector<int64_t> rz(d + 1);
  rz[0] = 1;
  rz[d] = 1;
  for (int64_t i = 1; i < d; ++i) {
    rz[i] = fixed_rank ? als_residual_rank : (kickrank + kick2);
  }
  std::vector<torch::Tensor> z_cores(d);
  if (!enrichment_disabled) {
    for (int64_t i = 0; i < d; ++i) {
      z_cores[i] = torch::randn({rz[i], N[i], rz[i + 1]}, options);
    }
    // Right-to-left orthogonalize z_cores.
    for (int64_t k = d - 1; k > 0; --k) {
      torch::Tensor core = z_cores[k].reshape({rz[k], -1}).t();
      auto [Q, R] = orthogonalize_maybe_qless(core, tsqr_block_size, use_qless_tsqr);
      rz[k] = Q.size(1);
      z_cores[k] = Q.t().reshape({rz[k], N[k], rz[k + 1]});
      z_cores[k - 1] = torch::tensordot(z_cores[k - 1], R.t(), {2}, {0});
    }
  }

  std::vector<torch::Tensor> Phiz(d + 1), Phiz_b(d + 1), Phis(d + 1),
    Phis_b(d + 1);
  Phiz[0] = torch::ones({1, 1, 1}, options);
  Phiz_b[0] = torch::ones({1, 1}, options);
  Phis[0] = torch::ones({1, 1, 1}, options);
  Phis_b[0] = torch::ones({1, 1}, options);
  Phiz[d] = torch::ones({1, 1, 1}, options);
  Phiz_b[d] = torch::ones({1, 1}, options);
  Phis[d] = torch::ones({1, 1, 1}, options);
  Phis_b[d] = torch::ones({1, 1}, options);

  std::vector<double> normA(d, 1.0), normb(d, 1.0), normx(d, 1.0);
  double nrmsc = 1.0;
  const double damp = 2.0;
  bool last = false;

  if (verbose) {
    std::cout << "Starting native AMEn solve with: eps=" << eps
               << ", max_rank=" << max_rank << ", nswp=" << nswp
               << ", kickrank=" << kickrank << ", kick2=" << kick2
               << (enrichment_disabled ? " (enrichment disabled, pure ALS)" : "")
               << std::endl;
  }

  for (int swp = 0; swp < nswp; ++swp) {
    // ================= Backward half-sweep =================
    for (int64_t k = d - 1; k > 0; --k) {
      if (!last && !enrichment_disabled) {
        torch::Tensor cz_new;
        // AMEnEnrichmentMode::SIMPLIFIED skips this backward-half-sweep
        // residual recomputation entirely (always falls through to reusing
        // the existing z_cores[k] below, same as the swp==0 case) -- the
        // only place it computes a residual at all is the forward
        // half-sweep, immediately after a core is actually re-solved (see
        // below), matching its doc comment ("compute the local residual
        // directly from the just-solved subproblem, skipping the full
        // global-residual contraction"). FULL and ALS_FIXED_RANK both
        // recompute a fresh (Phiz-projected) residual estimate here too,
        // even though the core at this position hasn't been re-solved yet
        // this sweep -- SIMPLIFIED trades that stale-but-fresh-looking
        // extra estimate for cheaper backward passes.
        if (swp > 0 && mode != AMEnEnrichmentMode::SIMPLIFIED) {
          torch::Tensor czA =
            FoldedLocalOperator::build(Phiz[k], A_cores[k], Phiz[k + 1])
              .apply(x_cores[k]);
          torch::Tensor czy = torch::tensordot(Phiz_b[k], b_cores[k], {0}, {0});
          czy = torch::tensordot(czy, Phiz_b[k + 1], {2}, {0});
          czy = czy * nrmsc - czA;
          torch::Tensor residual = czy.reshape({czy.size(0), -1});
          if (fixed_rank) {
            cz_new = fixed_rank_basis(residual.t(), als_residual_rank,
              tsqr_block_size, use_qless_tsqr, options);
          } else {
            auto [U, S, Vh] = torch::linalg_svd(residual, false);
            int64_t temp = std::min<int64_t>(kickrank, Vh.size(0));
            cz_new = Vh.index({Slice(0, temp), Ellipsis}).t();
            if (k < d - 1) {
              cz_new = torch::cat(
                {cz_new, torch::randn({cz_new.size(0), kick2}, options)}, 1);
            }
          }
        } else {
          cz_new = z_cores[k].reshape({rz[k], -1}).t();
        }
        auto [Qz, Rz] = orthogonalize_maybe_qless(cz_new, tsqr_block_size, use_qless_tsqr);
        rz[k] = Qz.size(1);
        z_cores[k] = Qz.t().reshape({rz[k], N[k], rz[k + 1]});
      }

      if (swp > 0) {
        nrmsc = nrmsc * normA[k - 1] * normx[k - 1] / normb[k - 1];
      }

      torch::Tensor core = x_cores[k].reshape({rx[k], N[k] * rx[k + 1]}).t();
      auto [Qx, Rx] = orthogonalize_maybe_qless(core, tsqr_block_size, use_qless_tsqr);
      torch::Tensor core_prev =
        torch::tensordot(x_cores[k - 1], Rx.t(), {2}, {0});
      rx[k] = Qx.size(1);

      double current_norm = core_prev.norm().item<double>();
      if (current_norm > 0) {
        core_prev = core_prev / current_norm;
      } else {
        current_norm = 1.0;
      }
      normx[k - 1] = normx[k - 1] * current_norm;

      x_cores[k] = Qx.t().reshape({rx[k], N[k], rx[k + 1]}).clone();
      x_cores[k - 1] = core_prev.clone();

      Phis[k] =
        compute_phi_bck_A(Phis[k + 1], x_cores[k], A_cores[k], x_cores[k]);
      Phis_b[k] = compute_phi_bck_rhs(Phis_b[k + 1], b_cores[k], x_cores[k]);

      double normA_k = Phis[k].norm().item<double>();
      normA_k = normA_k > 0 ? normA_k : 1.0;
      normA[k - 1] = normA_k;
      Phis[k] = Phis[k] / normA_k;

      double normb_k = Phis_b[k].norm().item<double>();
      normb_k = normb_k > 0 ? normb_k : 1.0;
      normb[k - 1] = normb_k;
      Phis_b[k] = Phis_b[k] / normb_k;

      nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1]);

      if (!last && !enrichment_disabled) {
        Phiz[k] =
          compute_phi_bck_A(Phiz[k + 1], z_cores[k], A_cores[k], x_cores[k]) /
          normA[k - 1];
        Phiz_b[k] =
          compute_phi_bck_rhs(Phiz_b[k + 1], b_cores[k], z_cores[k]) /
          normb[k - 1];
      }
    }

    // ================= Forward half-sweep =================
    double max_res = 0.0;
    for (int64_t k = 0; k < d; ++k) {
      torch::Tensor previous_solution = x_cores[k].reshape({-1, 1});

      torch::Tensor rhs = torch::tensordot(Phis_b[k], b_cores[k] * nrmsc, {0}, {0});
      rhs = torch::tensordot(rhs, Phis_b[k + 1], {2}, {0}).reshape({-1, 1});
      double norm_rhs = rhs.norm().item<double>();

      double real_tol = (eps / std::sqrt(static_cast<double>(d))) / damp;

      bool use_full = rx[k] * N[k] * rx[k + 1] < max_full;
      torch::Tensor solution_now;
      double res_old, res_new;
      // Whether this core's local solve target was deliberately loosened by
      // the forcing term (see below). When true, `res_new` reflects
      // accuracy left on the table on purpose, not genuine local-solve
      // difficulty, so it must NOT be used as a rank-truncation floor below
      // (see the `truncation_floor` computation) -- doing so previously
      // caused the adaptive rank-truncation loop to discard rank the
      // solution actually needs, stalling convergence (confirmed via a real
      // 2D neutron-transport regression case).
      bool was_forced_loose = false;

      FoldedLocalOperator local_op =
        FoldedLocalOperator::build(Phis[k], A_cores[k], Phis[k + 1]);
      torch::Tensor B_dense;

      if (use_full) {
        B_dense = local_op.to_dense();
        solution_now = torch::linalg_solve(B_dense, rhs);
        res_old = (torch::matmul(B_dense, previous_solution) - rhs).norm().item<double>() /
          norm_rhs;
        res_new =
          (torch::matmul(B_dense, solution_now) - rhs).norm().item<double>() / norm_rhs;
      } else {
        torch::Tensor rhs_shaped = rhs.reshape({rx[k], N[k], rx[k + 1]});
        torch::Tensor prev_shaped = previous_solution.reshape({rx[k], N[k], rx[k + 1]});

        res_old =
          (local_op.apply(prev_shaped).reshape({-1, 1}) - rhs).norm().item<double>() /
          norm_rhs;

        // Forcing term (inexact-Newton style, matching the local-tolerance
        // heuristic of Roehrig-Zoellner et al. 2025 / pitts' "AMEn+ALS"):
        // don't ask GMRES for more accuracy than the current local residual
        // already warrants. Early sweeps (res_old large, far from
        // convergence) get a loose target -- no point polishing a core that
        // will be overwritten by enrichment/further sweeps anyway; the
        // target tightens toward `real_tol` as res_old shrinks, so the
        // final sweep still gets full accuracy.
        double forcing_tol = use_local_forcing
          ? std::max(real_tol, std::min(gmres_forcing_ceiling, res_old))
          : real_tol;
        was_forced_loose = forcing_tol > real_tol;

        // Local (per-core) preconditioning: solve `(A M^-1) y = rhs`
        // directly (M is the cheap Jacobi-style block built from the
        // diagonal of Phis[k]/Phis[k+1], see local_preconditioner.hpp) with
        // `y0 = M x0`, then recover `x = M^-1 y`. Only on the iterative
        // path -- there's nothing to precondition when solving exactly.
        torch::Tensor sol_shaped;
        bool preconditioned_ok = false;
        if (preconditioner != AMEnPreconditioner::NONE) {
          // The Jacobi-style block this preconditioner is built from can be
          // genuinely singular for a particular core/problem (confirmed:
          // the torchTT baseline hits the exact same LAPACK/cuSOLVER
          // pivoting failure on the same input) -- fall back to the
          // unpreconditioned solve for just this core rather than letting a
          // rare degenerate block abort the whole sweep.
          try {
            LocalPreconditioner prec = LocalPreconditioner::build(
              Phis[k], A_cores[k], Phis[k + 1], preconditioner);
            torch::Tensor y0 = prec.apply_forward(prev_shaped);
            torch::Tensor sol_y = gmres_solve(local_op, rhs_shaped, y0,
              local_iterations, resets, forcing_tol, !use_gpu_batched_gmres, &prec);
            sol_shaped = prec.apply_inverse(sol_y);
            preconditioned_ok = true;
          } catch (const c10::Error&) {
            preconditioned_ok = false;
          }
        }
        if (!preconditioned_ok) {
          sol_shaped = gmres_solve(local_op, rhs_shaped, prev_shaped,
            local_iterations, resets, forcing_tol, !use_gpu_batched_gmres);
        }
        solution_now = sol_shaped.reshape({-1, 1});
        res_new =
          (local_op.apply(sol_shaped).reshape({-1, 1}) - rhs).norm().item<double>() /
          norm_rhs;
      }

      max_res = std::max(max_res, res_old);

      solution_now = solution_now.reshape({rx[k] * N[k], rx[k + 1]});

      torch::Tensor u, s, v;
      int64_t r;

      if (k < d - 1) {
        auto [U, S, Vh] = torch::linalg_svd(solution_now, false);
        u = U;
        s = S;
        v = Vh;
        r = u.size(1);
        // See `was_forced_loose` above: when this core's solve was
        // deliberately under-solved for speed, `res_new` doesn't reflect
        // genuine local-solve difficulty, so don't let it loosen the
        // truncation floor -- fall back to the tight per-core target alone.
        double truncation_floor =
          was_forced_loose ? real_tol * damp : std::max(res_new, real_tol * damp);
        while (r > 0) {
          torch::Tensor sol_r = torch::matmul(
            u.index({Ellipsis, Slice(0, r)}) * s.index({Slice(0, r)}),
            v.index({Slice(0, r), Ellipsis}));
          double res;
          if (use_full) {
            res = (torch::matmul(B_dense, sol_r.reshape({-1, 1})) - rhs)
                    .norm()
                    .item<double>() /
              norm_rhs;
          } else {
            res = (local_op.apply(sol_r.reshape({rx[k], N[k], rx[k + 1]}))
                     .reshape({-1, 1}) -
                    rhs)
                    .norm()
                    .item<double>() /
              norm_rhs;
          }
          if (res > truncation_floor) {
            break;
          }
          --r;
        }
        ++r;
        r = (r < u.size(1) && r < max_rank) ? r : std::min<int64_t>(u.size(1), max_rank);
      } else {
        auto [Q, R] = orthogonalize_maybe_qless(solution_now, tsqr_block_size, use_qless_tsqr);
        u = Q;
        v = R;
        r = u.size(1);
        s = torch::ones({r}, options);
      }

      u = u.index({Ellipsis, Slice(0, r)});
      torch::Tensor tmp1 = torch::diag(s.index({Slice(0, r)}));
      torch::Tensor tmp2 = v.index({Slice(0, r), Ellipsis});
      v = torch::matmul(tmp1, tmp2).t();

      if (!last && !enrichment_disabled) {
        torch::Tensor tmp = torch::matmul(u, v.t()).reshape({rx[k], N[k], rx[k + 1]});
        torch::Tensor czA =
          FoldedLocalOperator::build(Phiz[k], A_cores[k], Phiz[k + 1])
            .apply(tmp);
        torch::Tensor czy = torch::tensordot(Phiz_b[k], nrmsc * b_cores[k], {0}, {0});
        czy = torch::tensordot(czy, Phiz_b[k + 1], {2}, {0});
        torch::Tensor tmp_z = (czy - czA).reshape({rz[k] * N[k], rz[k + 1]});

        torch::Tensor tmp3;
        if (fixed_rank) {
          // At the right TT boundary (k == d-1), rz[k+1] == rz[d] must stay
          // exactly 1 -- don't pad tmp_z (whose width is rz[k+1]_old) up to
          // als_residual_rank there, matching how FULL mode's SVD is
          // naturally capped by the same dimensional constraint (no kick2
          // augmentation at k == d-1 either).
          int64_t target_rank =
            (k < d - 1) ? als_residual_rank : std::min(als_residual_rank, tmp_z.size(1));
          tmp3 = fixed_rank_basis(
            tmp_z, target_rank, tsqr_block_size, use_qless_tsqr, options);
        } else {
          auto [Uz, Sz, Vhz] = torch::linalg_svd(tmp_z, false);
          int64_t rtmp = std::min<int64_t>(kickrank, Uz.size(1));
          tmp3 = Uz.index({Ellipsis, Slice(0, rtmp)});
          if (k < d - 1) {
            tmp3 =
              torch::cat({tmp3, torch::randn({tmp3.size(0), kick2}, options)}, 1);
          }
        }
        auto [Qz2, Rz2] = orthogonalize_maybe_qless(tmp3, tsqr_block_size, use_qless_tsqr);
        rz[k + 1] = Qz2.size(1);
        z_cores[k] = Qz2.reshape({rz[k], N[k], rz[k + 1]}).clone();
      }

      if (k < d - 1) {
        if (!last && !enrichment_disabled) {
          torch::Tensor tmp = torch::matmul(u, v.t()).reshape({rx[k], N[k], rx[k + 1]});
          torch::Tensor left_res =
            FoldedLocalOperator::build(Phis[k], A_cores[k], Phiz[k + 1])
              .apply(tmp);
          torch::Tensor left_b = torch::tensordot(Phis_b[k], b_cores[k] * nrmsc, {0}, {0});
          left_b = torch::tensordot(left_b, Phiz_b[k + 1], {2}, {0});

          torch::Tensor uk = (left_b - left_res).reshape({u.size(0), -1});
          uk = torch::cat({u, uk}, 1);
          int64_t r_add = left_res.size(2);

          auto [Uk2, Rmat] = orthogonalize_maybe_qless(uk, tsqr_block_size, use_qless_tsqr);
          u = Uk2;

          torch::Tensor toadd = torch::zeros({rx[k + 1], r_add}, options);
          v = torch::cat({v, toadd}, 1);
          v = torch::matmul(v, Rmat.t());
        }

        r = u.size(1);
        v = torch::tensordot(v, x_cores[k + 1], {0}, {0});

        nrmsc = nrmsc * normA[k] * normx[k] / normb[k];

        double norm_now = v.norm().item<double>();
        if (norm_now > 0) {
          v = v / norm_now;
        } else {
          norm_now = 1.0;
        }
        normx[k] = normx[k] * norm_now;

        x_cores[k] = u.reshape({rx[k], N[k], r}).clone();
        x_cores[k + 1] = v.reshape({r, N[k + 1], rx[k + 2]}).clone();
        rx[k + 1] = r;

        Phis[k + 1] =
          compute_phi_fwd_A(Phis[k], x_cores[k], A_cores[k], x_cores[k]);
        Phis_b[k + 1] = compute_phi_fwd_rhs(Phis_b[k], b_cores[k], x_cores[k]);

        double normA_k = Phis[k + 1].norm().item<double>();
        normA_k = normA_k > 0 ? normA_k : 1.0;
        normA[k] = normA_k;
        Phis[k + 1] = Phis[k + 1] / normA_k;

        double normb_k = Phis_b[k + 1].norm().item<double>();
        normb_k = normb_k > 0 ? normb_k : 1.0;
        normb[k] = normb_k;
        Phis_b[k + 1] = Phis_b[k + 1] / normb_k;

        nrmsc = nrmsc * normb[k] / (normA[k] * normx[k]);

        if (!last && !enrichment_disabled) {
          Phiz[k + 1] =
            compute_phi_fwd_A(Phiz[k], z_cores[k], A_cores[k], x_cores[k]) /
            normA[k];
          Phiz_b[k + 1] =
            compute_phi_fwd_rhs(Phiz_b[k], b_cores[k], z_cores[k]) / normb[k];
        }
      } else {
        torch::Tensor usv = torch::matmul(
          u * s.index({Slice(0, r)}), v.index({Slice(0, r), Ellipsis}).t());
        x_cores[k] = usv.reshape({rx[k], N[k], rx[k + 1]});
      }
    }

    if (verbose) {
      std::cout << "Sweep " << (swp + 1) << "/" << nswp
                 << (last ? " (last, polishing)" : "") << ": max_res=" << max_res
                 << ", ranks=[ ";
      for (int64_t rr : rx) {
        std::cout << rr << " ";
      }
      std::cout << "]" << std::endl;
    }

    if (last) {
      break;
    }
    if (max_res < eps) {
      last = true;
    }
  }

  double norm_x = 0.0;
  for (int64_t i = 0; i < d - 1; ++i) {
    norm_x += std::log(normx[i]);
  }
  norm_x = std::exp(norm_x / static_cast<double>(d));
  for (int64_t i = 0; i < d; ++i) {
    x_cores[i] = x_cores[i] * norm_x;
  }

  return x_cores;
}

TTEngine amen_solve_dispatch(const TTEngine& A, const TTEngine& b,
  std::optional<TTEngine> x0, int nswp, double eps, int max_rank,
  int max_full, int kickrank, int kick2, int local_iterations, int resets,
  bool verbose, AMEnPreconditioner preconditioner,
  const AMEnNativeOptions& native_opts, const char* caller)
{
  std::vector<torch::Tensor> A_cores(A.get_cores().begin(), A.get_cores().end());
  std::vector<torch::Tensor> b_cores_4d(b.get_cores().begin(), b.get_cores().end());

  if (A_cores.size() != b_cores_4d.size()) {
    throw utils::runtime_error(caller, "`A` and `b` must have the same number of cores");
  }

  const int64_t d = static_cast<int64_t>(A_cores.size());
  std::vector<int64_t> N(d);
  for (int64_t i = 0; i < d; ++i) {
    if (A_cores[i].size(1) != b_cores_4d[i].size(1)) {
      throw utils::runtime_error(
        caller, "The `m_modes` of `A` must equal the `m_modes` of `b`");
    }
    if (b_cores_4d[i].size(2) != 1) {
      throw utils::runtime_error(
        caller, "`b` must be a TT-vector with all `m_modes` equal to 1");
    }
    if (A_cores[i].size(1) != A_cores[i].size(2)) {
      throw utils::runtime_error(caller,
        "The native backend requires a square operator per core (`m_modes` "
        "== `n_modes`)");
    }
    N[i] = A_cores[i].size(2);
  }

  std::vector<torch::Tensor> b_cores(d);
  for (int64_t i = 0; i < d; ++i) {
    b_cores[i] = b_cores_4d[i].squeeze(2);
  }

  std::optional<std::vector<torch::Tensor>> x0_cores;
  if (x0.has_value()) {
    x0_cores = std::vector<torch::Tensor>();
    x0_cores->reserve(d);
    for (const auto& core : x0->get_cores()) {
      if (core.size(2) != 1) {
        throw utils::runtime_error(
          caller, "`x0` must be a TT-vector with all `m_modes` equal to 1");
      }
      x0_cores->push_back(core.squeeze(2));
    }
  }

  // Rank1Preconditioner::apply_left/apply_right/apply_right_inverse/
  // sandwich_operator all use the plain 3-D vector-core convention (matching
  // the sweep driver's own internal representation, below) -- precondition
  // only after squeezing `b`/`x0` down to that convention. A user-supplied
  // `x0` (already living in the operator's input space) is mapped into the
  // preconditioned/canonical space via `apply_right_inverse` so warm-starting
  // still works under preconditioning, instead of being discarded.
  std::optional<Rank1Preconditioner> prec;
  if (preconditioner == AMEnPreconditioner::RANK1) {
    prec = Rank1Preconditioner::build(A);
    A_cores = prec->sandwich_operator(A_cores);
    b_cores = prec->apply_left(b_cores);
    if (x0_cores.has_value()) {
      x0_cores = prec->apply_right_inverse(*x0_cores);
    }
  }
  // RANK1 is fully handled above (a global sandwich, not a per-core local
  // preconditioner) -- amen_sweep only understands NONE/LOCAL_C_PREC/
  // LOCAL_R_PREC, so it would reject RANK1 if passed through.
  const AMEnPreconditioner local_preconditioner =
    (preconditioner == AMEnPreconditioner::RANK1) ? AMEnPreconditioner::NONE
                                                    : preconditioner;

  std::vector<torch::Tensor> x_cores;
  if (x0_cores.has_value()) {
    x_cores = std::move(*x0_cores);
  } else {
    for (int64_t i = 0; i < d; ++i) {
      x_cores.push_back(torch::ones({1, N[i], 1}, A_cores[0].options()));
    }
  }

  std::vector<torch::Tensor> result = amen_sweep(A_cores, b_cores, x_cores, N,
    nswp, eps, max_rank, max_full, kickrank, kick2, local_iterations, resets,
    verbose, local_preconditioner, native_opts.tsqr_block_size,
    native_opts.use_qless_tsqr, native_opts.enrichment_mode,
    native_opts.als_residual_rank, native_opts.use_local_forcing,
    native_opts.gmres_forcing_ceiling, native_opts.use_gpu_batched_gmres);

  if (prec.has_value()) {
    result = prec->apply_right(result);
  }

  return TTEngine(TTEngine::Tensors(result.cbegin(), result.cend()), true);
}

} // namespace ttnte::linalg::amen
