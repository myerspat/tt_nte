#include "ttnte/linalg/amen/local_preconditioner.hpp"

namespace ttnte::linalg::amen {

LocalPreconditioner LocalPreconditioner::build(const torch::Tensor& phi_left,
  const torch::Tensor& a_core, const torch::Tensor& phi_right,
  AMEnPreconditioner mode)
{
  TORCH_CHECK(mode == AMEnPreconditioner::LOCAL_C_PREC ||
                mode == AMEnPreconditioner::LOCAL_R_PREC,
    "LocalPreconditioner::build expects mode LOCAL_C_PREC or LOCAL_R_PREC");
  TORCH_CHECK(phi_left.dim() == 3 && phi_left.size(0) == phi_left.size(2),
    "LocalPreconditioner requires a square local operator (phi_left's l == r)");
  TORCH_CHECK(phi_right.dim() == 3 && phi_right.size(0) == phi_right.size(2),
    "LocalPreconditioner requires a square local operator (phi_right's L == "
    "R)");
  TORCH_CHECK(a_core.dim() == 4 && a_core.size(1) == a_core.size(2),
    "LocalPreconditioner requires a square local operator (a_core's m == n)");

  const int64_t d = phi_left.size(0);
  const int64_t d2 = phi_right.size(0);
  const int64_t mode_size = a_core.size(1);

  // Jl = sum_s diag(phi_left)[s, d] * a_core[s, m, n, S] -> [d, m, n, S].
  // Only the LEFT interface is diagonalized in both modes -- the cheap,
  // Jacobi-style approximation this preconditioner is built on.
  torch::Tensor diag_left = torch::diagonal(phi_left, 0, 0, 2); // [s, d]
  torch::Tensor Jl =
    torch::tensordot(diag_left, a_core, {0}, {0}); // [d, m, n, S]

  LocalPreconditioner prec;
  prec.mode_ = mode;
  prec.r_ = d;
  prec.n_ = mode_size;

  if (mode == AMEnPreconditioner::LOCAL_C_PREC) {
    // LOCAL_C_PREC: also diagonalize the right interface -- cheapest option,
    // one small n x n block per (r, R) rank-index pair.
    torch::Tensor diag_right = torch::diagonal(phi_right, 0, 0, 2); // [S, d2]
    torch::Tensor Jt =
      torch::tensordot(Jl, diag_right, {3}, {0});     // [d, m, n, d2]
    prec.Jt_ = Jt.permute({0, 3, 1, 2}).contiguous(); // [d, d2, m, n]
    prec.R_ = d2;
  } else {
    // LOCAL_R_PREC: keep the right interface's full (non-diagonal)
    // structure -- a bigger, more accurate block, batched only over the
    // left rank index.
    torch::Tensor Jt =
      torch::tensordot(Jl, phi_right, {3}, {1});   // [d, m, n, L, R]
    Jt = Jt.permute({0, 1, 3, 2, 4}).contiguous(); // [d, m, L, n, R]
    prec.R_ = phi_right.size(2);
    prec.Jt_ = Jt.reshape({d, mode_size * prec.R_, mode_size * prec.R_});
  }
  return prec;
}

torch::Tensor LocalPreconditioner::apply_forward(const torch::Tensor& x) const
{
  if (mode_ == AMEnPreconditioner::LOCAL_C_PREC) {
    torch::Tensor x_p = x.permute({0, 2, 1}).unsqueeze(-1);  // [r, R, n, 1]
    torch::Tensor out = torch::matmul(Jt_, x_p).squeeze(-1); // [r, R, n]
    return out.permute({0, 2, 1}).contiguous();              // [r, n, R]
  }
  torch::Tensor x_flat = x.reshape({r_, n_ * R_, 1});
  torch::Tensor out = torch::matmul(Jt_, x_flat); // [r, n*R, 1]
  return out.reshape({r_, n_, R_});
}

torch::Tensor LocalPreconditioner::apply_inverse(const torch::Tensor& y) const
{
  if (mode_ == AMEnPreconditioner::LOCAL_C_PREC) {
    torch::Tensor y_p = y.permute({0, 2, 1}).unsqueeze(-1); // [r, R, n, 1]
    torch::Tensor out = torch::linalg_solve(Jt_, y_p).squeeze(-1); // [r, R, n]
    return out.permute({0, 2, 1}).contiguous();                    // [r, n, R]
  }
  torch::Tensor y_flat = y.reshape({r_, n_ * R_, 1});
  torch::Tensor out = torch::linalg_solve(Jt_, y_flat); // [r, n*R, 1]
  return out.reshape({r_, n_, R_});
}

} // namespace ttnte::linalg::amen
