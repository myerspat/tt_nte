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

torch::Tensor gmres_solve(const FoldedLocalOperator& op, const torch::Tensor& rhs,
  const torch::Tensor& x0, int max_iterations, int restarts, double rel_tol,
  bool prefer_incremental, const LocalPreconditioner* prec)
{
  if (rhs.is_cuda() && !prefer_incremental) {
    return gmres_solve_gpu(op, rhs, x0, max_iterations, restarts, rel_tol, prec);
  }
  return gmres_solve_cpu(op, rhs, x0, max_iterations, restarts, rel_tol, prec);
}

torch::Tensor gmres_solve_cpu(const FoldedLocalOperator& op,
  const torch::Tensor& rhs, const torch::Tensor& x0, int max_iterations,
  int restarts, double rel_tol, const LocalPreconditioner* prec)
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

  const int64_t m = std::max<int64_t>(1, std::min<int64_t>(max_iterations, dim));

  for (int restart = 0; restart < restarts; ++restart) {
    torch::Tensor r = b - local_apply(x);
    double beta0 = r.norm().item<double>();
    if (beta0 <= rel_tol * b_norm) {
      break;
    }

    torch::Tensor V = torch::zeros({dim, m + 1}, options);
    V.index_put_({Slice(), 0}, r / beta0);

    // Upper-triangularized Hessenberg (via incremental Givens rotations).
    torch::Tensor R = torch::zeros({m + 1, m}, options);
    torch::Tensor beta = torch::zeros({m + 1}, options);
    beta.index_put_({0}, beta0);

#ifdef USE_CUDA
    const bool on_cuda = rhs.is_cuda();
#else
    const bool on_cuda = false;
#endif

    // On CUDA, cs/sn stay device-resident and the whole rotation update runs
    // as a single fused kernel (see below) -- avoids the O(k) host round
    // trips a per-element host loop would otherwise force per iteration. On
    // CPU, a plain host loop (unchanged from the original implementation) is
    // simplest and already cheap (no device round trips to avoid).
    std::vector<double> cs, sn;
    torch::Tensor cs_t, sn_t;
    if (on_cuda) {
#ifdef USE_CUDA
      cs_t = torch::ones({m}, options);
      sn_t = torch::zeros({m}, options);
#endif
    } else {
      cs.assign(m, 1.0);
      sn.assign(m, 0.0);
    }

    int64_t k_used = 0;
    for (int64_t k = 0; k < m; ++k) {
      torch::Tensor v = local_apply(V.index({Slice(), k}));

      // Modified Gram-Schmidt with one reorthogonalization pass.
      torch::Tensor Vk = V.index({Slice(), Slice(0, k + 1)});
      torch::Tensor h = torch::mv(Vk.transpose(0, 1), v);
      v = v - torch::mv(Vk, h);
      torch::Tensor h2 = torch::mv(Vk.transpose(0, 1), v);
      v = v - torch::mv(Vk, h2);
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
        // the magnitude of beta[k+1] is needed, never its sign).
        torch::Tensor check =
          torch::stack({vnorm_t, beta.index({k + 1}).abs()}).cpu();
        vnorm = check[0].item<double>();
        abs_new_beta_kp1 = check[1].item<double>();
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
    x = x + torch::matmul(V_used, y).reshape(x.sizes());
  }

  return x;
}

torch::Tensor gmres_solve_gpu(const FoldedLocalOperator& op,
  const torch::Tensor& rhs, const torch::Tensor& x0, int max_iterations,
  int restarts, double rel_tol, const LocalPreconditioner* prec)
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

  auto local_apply = [&](const torch::Tensor& v) -> torch::Tensor {
    torch::Tensor v_shaped = v.reshape(rhs.sizes());
    if (prec != nullptr) {
      v_shaped = prec->apply_inverse(v_shaped);
    }
    return op.apply(v_shaped).reshape(-1);
  };

  const int64_t m = std::max<int64_t>(1, std::min<int64_t>(max_iterations, dim));

  for (int restart = 0; restart < restarts; ++restart) {
    torch::Tensor r = b - local_apply(x);
    torch::Tensor beta0_t = r.norm();
    // One host sync per restart (not per iteration) to decide whether
    // another restart is needed.
    double beta0 = beta0_t.item<double>();
    if (beta0 <= rel_tol * b_norm) {
      break;
    }

    torch::Tensor V = torch::zeros({dim, m + 1}, options);
    V.index_put_({Slice(), 0}, r / beta0_t);
    torch::Tensor H = torch::zeros({m + 1, m}, options);

    // Fixed-budget Arnoldi expansion: no incremental least-squares
    // tracking, no per-iteration host sync.
    for (int64_t k = 0; k < m; ++k) {
      torch::Tensor v = local_apply(V.index({Slice(), k}));

      torch::Tensor Vk = V.index({Slice(), Slice(0, k + 1)});
      torch::Tensor h = torch::mv(Vk.transpose(0, 1), v);
      v = v - torch::mv(Vk, h);
      torch::Tensor h2 = torch::mv(Vk.transpose(0, 1), v);
      v = v - torch::mv(Vk, h2);
      h = h + h2;

      torch::Tensor vnorm = v.norm();
      H.index_put_({Slice(0, k + 1), k}, h);
      H.index_put_({k + 1, k}, vnorm);

      // Divide-by-(near)zero on breakdown just yields a garbage column that
      // contributes ~nothing after the least-squares solve below -- robust
      // without a data-dependent branch (and so without a host sync).
      V.index_put_({Slice(), k + 1}, v / torch::clamp_min(vnorm, 1e-300));
    }

    torch::Tensor beta_vec = torch::zeros({m + 1, 1}, options);
    beta_vec.index_put_({0, 0}, beta0_t);

    // Single small least-squares solve per restart, via LAPACK/cuSOLVER
    // (QR/SVD-based internally) rather than normal equations -- avoids
    // squaring the condition number of H.
    auto lstsq_result = torch::linalg_lstsq(H, beta_vec, std::nullopt, std::nullopt);
    torch::Tensor y = std::get<0>(lstsq_result);

    torch::Tensor V_used = V.index({Slice(), Slice(0, m)});
    x = x + torch::matmul(V_used, y).reshape(x.sizes());
  }

  return x;
}

} // namespace ttnte::linalg::amen
