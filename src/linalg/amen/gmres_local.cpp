#include "ttnte/linalg/amen/gmres_local.hpp"

#ifdef USE_CUDA
#include "ttnte/linalg/amen/gmres_kernels.cuh"
#endif

#include <cmath>

using namespace torch::indexing;

namespace ttnte::linalg::amen {

namespace {

void check_square(const FoldedLocalOperator& op)
{
  TORCH_CHECK(op.l() == op.r() && op.m() == op.n() && op.L() == op.R(),
    "gmres_solve requires a square local operator (domain shape must equal "
    "codomain shape)");
}

} // namespace

torch::Tensor gmres_solve(const FoldedLocalOperator& op,
  const torch::Tensor& rhs, const torch::Tensor& x0, int max_iterations,
  int restarts, double rel_tol, bool prefer_incremental,
  const LocalPreconditioner* prec, int check_interval,
  bool gmres_mixed_precision)
{
  if (rhs.is_cuda() && !prefer_incremental) {
    return gmres_solve_gpu(op, rhs, x0, max_iterations, restarts, rel_tol, prec,
      check_interval, gmres_mixed_precision);
  }
  return gmres_solve_cpu(op, rhs, x0, max_iterations, restarts, rel_tol, prec,
    gmres_mixed_precision);
}

torch::Tensor gmres_solve_cpu(const FoldedLocalOperator& op,
  const torch::Tensor& rhs, const torch::Tensor& x0, int max_iterations,
  int restarts, double rel_tol, const LocalPreconditioner* prec,
  bool gmres_mixed_precision)
{
  check_square(op);
  auto options = rhs.options();
  torch::Tensor x = x0.clone();
  torch::Tensor b = rhs.reshape(-1);
  const int64_t dim = b.numel();
  const double b_norm = b.norm().item<double>();
  if (b_norm == 0.0) {
    return torch::zeros_like(x0);
  }

  // Composes the local preconditioner's inverse into the operator apply
  // (`op.apply(prec->apply_inverse(v))` instead of `op.apply(v)`) at the
  // single point each strategy calls it -- reduces to the original
  // behavior exactly when `prec` is null.
  auto local_apply = [&](const torch::Tensor& v) -> torch::Tensor {
    torch::Tensor v_shaped = v.reshape(rhs.sizes());
    if (prec != nullptr) {
      v_shaped = prec->apply_inverse(v_shaped);
    }
    return op.apply(v_shaped).reshape(-1);
  };

  const int64_t m =
    std::max<int64_t>(1, std::min<int64_t>(max_iterations, dim));

#ifdef USE_CUDA
  const bool on_cuda = rhs.is_cuda();
#else
  const bool on_cuda = false;
#endif

  // AMEnNativeOptions::gmres_mixed_precision: applies to both the CUDA
  // (fused_givens_apply) and CPU (scalar host-loop Givens) strategy branches
  // below -- both read/write V/R/beta through plain dtype-agnostic ATen ops,
  // so lowering their dtype works the same way on either device. Only when
  // unpreconditioned -- `LocalPreconditioner` isn't precision-matched here,
  // so preconditioned calls always fall back to the unmixed (float64) path.
  // `op32`/`local_apply32` are only ever touched when `mixed` is true; when
  // `gmres_mixed_precision` is false (the default), every buffer below keeps
  // using `options` (rhs's own dtype) exactly as before this option existed.
  const bool mixed = gmres_mixed_precision && prec == nullptr;
  FoldedLocalOperator op32;
  if (mixed) {
    op32 = op.to_float32();
  }
  auto local_apply32 = [&](const torch::Tensor& v) -> torch::Tensor {
    return op32.apply(v.reshape(rhs.sizes())).reshape(-1);
  };
  torch::TensorOptions compute_options =
    mixed ? options.dtype(torch::kFloat32) : options;

  // Reused across restarts instead of reallocated each time -- every entry
  // that matters is overwritten before it's read within a given restart
  // (see the per-buffer notes below), so no full re-zero is needed between
  // restarts either, just the couple of entries seeded at each restart's
  // start.
  //   V: column 0 is reseeded every restart (below); columns 1..k_used-1
  //      are always written during that restart's k-loop before being
  //      included in `V_used` (a breakdown at column k+1 both skips writing
  //      it and excludes it from `V_used`, which only goes up to column
  //      k_used-1 = k -- see the CPU-branch `vnorm > 1e-300` guard below).
  //   R: `linalg_solve_triangular(upper=true)` only ever reads the
  //      upper-triangular entries this restart's k-loop itself wrote (rows
  //      0..k for column k); it never reads the lower triangle, so stale
  //      entries left there by a previous restart are harmless.
  //   beta: index 0 is reseeded every restart (below); index k+1 is always
  //      written at iteration k before being read (as "beta[k]") at
  //      iteration k+1.
  //   cs/sn (host) and cs_t/sn_t (device, on_cuda only): index k is always
  //      written at iteration k before being read at any iteration k' > k
  //      within the same restart (same self-healing argument as beta).
  torch::Tensor V = torch::zeros({dim, m + 1}, compute_options);
  torch::Tensor R = torch::zeros({m + 1, m}, compute_options);
  torch::Tensor beta = torch::zeros({m + 1}, compute_options);
  std::vector<double> cs(m, 1.0), sn(m, 0.0);
  torch::Tensor cs_t, sn_t, host_check;
#ifdef USE_CUDA
  if (on_cuda) {
    cs_t = torch::ones({m}, compute_options);
    sn_t = torch::zeros({m}, compute_options);
    // Reused D2H landing buffer for the per-iteration convergence check
    // below. Pinned so the transfer uses the CUDA DMA fast path instead of
    // the internally-staged, effectively-blocking copy a fresh `.cpu()`
    // allocation (default pageable memory) forces every call.
    host_check = torch::empty({2}, torch::TensorOptions()
                                     .dtype(compute_options.dtype())
                                     .pinned_memory(true));
  }
#endif

  for (int restart = 0; restart < restarts; ++restart) {
    torch::Tensor r = b - local_apply(x);
    double beta0 = r.norm().item<double>();
    if (beta0 <= rel_tol * b_norm) {
      break;
    }

    // `r` is always float64 (the true residual, computed above via the
    // original `op`); cast to `V`'s dtype (float32 when mixed) on the way in
    // -- a no-op when not mixed, since `V` is float64 then too.
    V.index_put_({Slice(), 0}, (r / beta0).to(V.scalar_type()));
    beta.index_put_({0}, beta0);

    int64_t k_used = 0;
    for (int64_t k = 0; k < m; ++k) {
      // Inner Krylov build uses the float32 operator when mixed; the true
      // residual above (and the next restart's) always uses the original.
      torch::Tensor v = mixed ? local_apply32(V.index({Slice(), k}))
                              : local_apply(V.index({Slice(), k}));

      // Modified Gram-Schmidt with one reorthogonalization pass. `addmv_`
      // fuses each "v -= Vk @ h" into the GEMV itself (one BLAS call
      // writing into v's own storage) instead of a separate mv() call plus
      // a standalone elementwise subtract -- same result, half the kernel
      // launches and allocations per pass.
      torch::Tensor Vk = V.index({Slice(), Slice(0, k + 1)});
      torch::Tensor h = torch::mv(Vk.transpose(0, 1), v);
      v.addmv_(Vk, h, /*beta=*/1.0, /*alpha=*/-1.0);
      torch::Tensor h2 = torch::mv(Vk.transpose(0, 1), v);
      v.addmv_(Vk, h2, /*beta=*/1.0, /*alpha=*/-1.0);
      h = h + h2;

      double vnorm;
      double abs_new_beta_kp1;

      if (on_cuda) {
#ifdef USE_CUDA
        torch::Tensor vnorm_t = v.norm();
        torch::Tensor r_col =
          cuda::fused_givens_apply(h, vnorm_t, cs_t, sn_t, beta, k);
        R.index_put_({Slice(0, k + 1), k}, r_col);
        V.index_put_({Slice(), k + 1}, v / torch::clamp_min(vnorm_t, 1e-300));

        // One combined host sync per iteration (down from O(k)) -- same
        // stopping condition as the CPU path below, just read together (only
        // the magnitude of beta[k+1] is needed, never its sign). Landing in
        // the reused pinned `host_check` (instead of a fresh `.cpu()` call)
        // avoids a pageable-memory D2H copy -- profiling on the C5G7 pincell
        // benchmark showed this exact transfer dominating wall clock via a
        // cudaMemcpyAsync spin-wait.
        host_check.copy_(torch::stack({vnorm_t, beta.index({k + 1}).abs()}));
        vnorm = host_check[0].item<double>();
        abs_new_beta_kp1 = host_check[1].item<double>();
#endif
      } else {
        vnorm = v.norm().item<double>();

        // Apply the previous k Givens rotations to the new Hessenberg
        // column.
        std::vector<double> h_vec(k + 2, 0.0);
        for (int64_t i = 0; i <= k; ++i) {
          h_vec[i] = h[i].item<double>();
        }
        h_vec[k + 1] = vnorm;
        for (int64_t i = 0; i < k; ++i) {
          double hi = h_vec[i], hip1 = h_vec[i + 1];
          h_vec[i] = cs[i] * hi + sn[i] * hip1;
          h_vec[i + 1] = -sn[i] * hi + cs[i] * hip1;
        }

        // New Givens rotation zeroing h_vec[k+1] against h_vec[k].
        double a = h_vec[k], bb = h_vec[k + 1];
        double denom = std::sqrt(a * a + bb * bb);
        double c = (denom > 0.0) ? a / denom : 1.0;
        double s = (denom > 0.0) ? bb / denom : 0.0;
        cs[k] = c;
        sn[k] = s;
        h_vec[k] = c * a + s * bb;

        for (int64_t i = 0; i <= k; ++i) {
          R.index_put_({i, k}, h_vec[i]);
        }

        double beta_k = beta[k].item<double>();
        double new_beta_k = c * beta_k;
        double new_beta_kp1 = -s * beta_k;
        beta.index_put_({k}, new_beta_k);
        beta.index_put_({k + 1}, new_beta_kp1);
        abs_new_beta_kp1 = std::abs(new_beta_kp1);

        if (vnorm > 1e-300) {
          V.index_put_({Slice(), k + 1}, v / vnorm);
        }
      }

      k_used = k + 1;

      if (abs_new_beta_kp1 <= rel_tol * b_norm || vnorm <= 1e-300) {
        break;
      }
    }

    torch::Tensor R_used = R.index({Slice(0, k_used), Slice(0, k_used)});
    torch::Tensor beta_used = beta.index({Slice(0, k_used)}).unsqueeze(1);
    torch::Tensor y = torch::linalg_solve_triangular(
      R_used, beta_used, /*upper=*/true, /*left=*/true);
    torch::Tensor V_used = V.index({Slice(), Slice(0, k_used)});
    // Cast the correction to x's own dtype before adding -- a no-op unless
    // mixed precision actually lowered V/R/beta's dtype below x's (e.g. a
    // no-op both in the unmixed float64 case and when the caller's own
    // rhs/x0 were already float32). The next restart's residual is always
    // recomputed via the original (unlowered) operator, so any
    // reduced-precision rounding error introduced here is corrected there
    // rather than accumulating.
    x = x + torch::matmul(V_used, y).to(x.scalar_type()).reshape(x.sizes());
  }

  return x;
}

torch::Tensor gmres_solve_gpu(const FoldedLocalOperator& op,
  const torch::Tensor& rhs, const torch::Tensor& x0, int max_iterations,
  int restarts, double rel_tol, const LocalPreconditioner* prec,
  int check_interval, bool gmres_mixed_precision)
{
  check_square(op);
  TORCH_CHECK(
    check_interval >= 1, "gmres_solve_gpu requires check_interval >= 1");
  auto options = rhs.options();
  torch::Tensor x = x0.clone();
  torch::Tensor b = rhs.reshape(-1);
  const int64_t dim = b.numel();
  const double b_norm = b.norm().item<double>();
  if (b_norm == 0.0) {
    return torch::zeros_like(x0);
  }

  auto local_apply = [&](const torch::Tensor& v) -> torch::Tensor {
    torch::Tensor v_shaped = v.reshape(rhs.sizes());
    if (prec != nullptr) {
      v_shaped = prec->apply_inverse(v_shaped);
    }
    return op.apply(v_shaped).reshape(-1);
  };

  const int64_t m =
    std::max<int64_t>(1, std::min<int64_t>(max_iterations, dim));

  // See gmres_solve_cpu's identical block above -- same mixed-precision
  // scheme (inner Krylov build in float32, true residual/solution update in
  // float64), just wired into this strategy's fixed-budget Arnoldi loop
  // instead of the incremental-Givens one.
  const bool mixed = gmres_mixed_precision && prec == nullptr;
  FoldedLocalOperator op32;
  if (mixed) {
    op32 = op.to_float32();
  }
  auto local_apply32 = [&](const torch::Tensor& v) -> torch::Tensor {
    return op32.apply(v.reshape(rhs.sizes())).reshape(-1);
  };
  torch::TensorOptions compute_options =
    mixed ? options.dtype(torch::kFloat32) : options;

  // Reused across restarts instead of reallocated each time.
  //   V: column 0 is reseeded every restart (below); columns 1..k_used-1 are
  //      always (unconditionally) written during that restart's k-loop
  //      before being included in `V_used`.
  //   beta_vec: index 0 is reseeded every restart; every other index is
  //      never written anywhere in this function (by design -- the Arnoldi
  //      least-squares RHS is `[beta0, 0, ..., 0]^T`), so it's already
  //      zero-valued forever once the initial allocation zeroed it, reused
  //      or not.
  //   H: unlike V/beta_vec, its below-subdiagonal entries must genuinely
  //      *be* zero (Hessenberg structure) for `linalg_lstsq` to see the
  //      right system -- `linalg_solve_triangular`'s "ignores what it
  //      doesn't need" argument (see `gmres_solve_cpu`) does not apply here.
  //      So `H` alone is explicitly re-zeroed every restart below; still
  //      cheaper than a fresh allocation.
  torch::Tensor V = torch::zeros({dim, m + 1}, compute_options);
  torch::Tensor H = torch::zeros({m + 1, m}, compute_options);
  torch::Tensor beta_vec = torch::zeros({m + 1, 1}, compute_options);

  for (int restart = 0; restart < restarts; ++restart) {
    torch::Tensor r = b - local_apply(x);
    torch::Tensor beta0_t = r.norm();
    // One host sync per restart (not per iteration) to decide whether
    // another restart is needed.
    double beta0 = beta0_t.item<double>();
    if (beta0 <= rel_tol * b_norm) {
      break;
    }

    // `r`/`beta0_t` are always float64 (the true residual, computed above
    // via the original `op`); cast to V/beta_vec's dtype (float32 when
    // mixed) on the way in -- a no-op when not mixed.
    V.index_put_({Slice(), 0}, (r / beta0_t).to(V.scalar_type()));
    H.zero_();
    beta_vec.index_put_({0, 0}, beta0_t.to(beta_vec.scalar_type()));

    // Fixed-budget Arnoldi expansion (no incremental Givens tracking), but
    // with a residual check amortized every `check_interval` iterations
    // instead of either 0 times (original behavior -- risks burning the
    // full `max_iterations` budget even long after convergence) or every
    // iteration (the CPU path's per-iteration host sync this strategy
    // exists to avoid). `y`/`k_used` carry the last-solved subspace so a
    // break (early or at the budget) never needs a final re-solve.
    int64_t k_used = m;
    torch::Tensor y;
    for (int64_t k = 0; k < m; ++k) {
      // Inner Krylov build uses the float32 operator when mixed; the true
      // residual above (and the next restart's) always uses the original.
      torch::Tensor v = mixed ? local_apply32(V.index({Slice(), k}))
                              : local_apply(V.index({Slice(), k}));

      // See gmres_solve_cpu's identical block above for why addmv_ is used
      // instead of a separate mv() + subtract.
      torch::Tensor Vk = V.index({Slice(), Slice(0, k + 1)});
      torch::Tensor h = torch::mv(Vk.transpose(0, 1), v);
      v.addmv_(Vk, h, /*beta=*/1.0, /*alpha=*/-1.0);
      torch::Tensor h2 = torch::mv(Vk.transpose(0, 1), v);
      v.addmv_(Vk, h2, /*beta=*/1.0, /*alpha=*/-1.0);
      h = h + h2;

      torch::Tensor vnorm = v.norm();
      H.index_put_({Slice(0, k + 1), k}, h);
      H.index_put_({k + 1, k}, vnorm);

      // Divide-by-(near)zero on breakdown just yields a garbage column that
      // contributes ~nothing after the least-squares solve below -- robust
      // without a data-dependent branch (and so without a host sync).
      V.index_put_({Slice(), k + 1}, v / torch::clamp_min(vnorm, 1e-300));

      const bool is_check_step =
        ((k + 1) % check_interval == 0) || (k == m - 1);
      if (!is_check_step) {
        continue;
      }

      torch::Tensor H_partial = H.index({Slice(0, k + 2), Slice(0, k + 1)});
      torch::Tensor beta_partial = beta_vec.index({Slice(0, k + 2)});
      auto lstsq_result = torch::linalg_lstsq(
        H_partial, beta_partial, std::nullopt, std::nullopt);
      torch::Tensor y_partial = std::get<0>(lstsq_result);
      torch::Tensor resid_norm =
        (beta_partial - torch::matmul(H_partial, y_partial)).norm();

      // One host sync per check (not per iteration).
      if (resid_norm.item<double>() <= rel_tol * b_norm) {
        y = y_partial;
        k_used = k + 1;
        break;
      }
      if (k == m - 1) {
        // Budget exhausted on this restart -- keep the full-subspace
        // solve computed above rather than solving again.
        y = y_partial;
        k_used = k + 1;
      }
    }

    torch::Tensor V_used = V.index({Slice(), Slice(0, k_used)});
    // Cast the correction to x's own dtype before adding -- see
    // gmres_solve_cpu's identical cast above for why.
    x = x + torch::matmul(V_used, y).to(x.scalar_type()).reshape(x.sizes());
  }

  return x;
}

} // namespace ttnte::linalg::amen
