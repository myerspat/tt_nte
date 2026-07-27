#include "ttnte/linalg/amen/local_operator.hpp"

namespace ttnte::linalg::amen {

FoldedLocalOperator FoldedLocalOperator::build(const torch::Tensor& phi_left,
  const torch::Tensor& a_core, const torch::Tensor& phi_right,
  double regularization)
{
  TORCH_CHECK(phi_left.dim() == 3, "phi_left must be 3-D [l, s, r]");
  TORCH_CHECK(a_core.dim() == 4, "a_core must be 4-D [s, m, n, S]");
  TORCH_CHECK(phi_right.dim() == 3, "phi_right must be 3-D [L, S, R]");

  FoldedLocalOperator op;
  op.l_ = phi_left.size(0);
  op.s_ = phi_left.size(1);
  op.r_ = phi_left.size(2);
  op.m_ = a_core.size(1);
  op.n_ = a_core.size(2);
  op.S_ = a_core.size(3);
  op.L_ = phi_right.size(0);
  op.R_ = phi_right.size(2);

  TORCH_CHECK(
    a_core.size(0) == op.s_, "a_core's left rank must match phi_left's A-rank");
  TORCH_CHECK(phi_right.size(1) == op.S_,
    "phi_right's A-rank must match a_core's right rank");

  op.phi_left_raw_ = phi_left.contiguous();
  op.a_core_raw_ = a_core.contiguous();
  op.phi_right_raw_ = phi_right.contiguous();

  // [l, s, r] -> [l*s, r]
  op.phi_left_mat_ = op.phi_left_raw_.reshape({op.l_ * op.s_, op.r_});
  // [s, m, n, S] -> [s, n, m, S] -> [s*n, m*S]
  op.a_core_mat_ = op.a_core_raw_.permute({0, 2, 1, 3})
                     .contiguous()
                     .reshape({op.s_ * op.n_, op.m_ * op.S_});
  // [L, S, R] -> [R, S, L] -> [R*S, L]
  op.phi_right_mat_ = op.phi_right_raw_.permute({2, 1, 0}).contiguous().reshape(
    {op.R_ * op.S_, op.L_});

  op.regularization_ = regularization;

  return op;
}

torch::Tensor FoldedLocalOperator::apply(const torch::Tensor& y) const
{
  torch::Tensor y_mat = y.reshape({r_, n_ * R_});

  // Step 1: [l*s, r] @ [r, n*R] -> [l*s, n*R] -> [l, s, n, R]
  torch::Tensor t1 =
    torch::matmul(phi_left_mat_, y_mat).reshape({l_, s_, n_, R_});
  // -> [l, R, s, n] -> [l*R, s*n]
  t1 = t1.permute({0, 3, 1, 2}).contiguous().reshape({l_ * R_, s_ * n_});

  // Step 2: [l*R, s*n] @ [s*n, m*S] -> [l*R, m*S] -> [l, R, m, S]
  torch::Tensor t2 = torch::matmul(t1, a_core_mat_).reshape({l_, R_, m_, S_});
  // -> [l, m, R, S] -> [l*m, R*S]
  t2 = t2.permute({0, 2, 1, 3}).contiguous().reshape({l_ * m_, R_ * S_});

  // Step 3: [l*m, R*S] @ [R*S, L] -> [l*m, L] -> [l, m, L]
  torch::Tensor out = torch::matmul(t2, phi_right_mat_).reshape({l_, m_, L_});
  if (regularization_ != 0.0) {
    out = out + regularization_ * y.reshape({l_, m_, L_});
  }
  return out;
}

torch::Tensor FoldedLocalOperator::to_dense() const
{
  // C[l, r, m, n, S] = sum_s phi_left[l, s, r] * a_core[s, m, n, S]
  torch::Tensor phi_left_lrs =
    phi_left_raw_.permute({0, 2, 1}).contiguous().reshape({l_ * r_, s_});
  torch::Tensor a_core_s_mnS = a_core_raw_.reshape({s_, m_ * n_ * S_});
  torch::Tensor C =
    torch::matmul(phi_left_lrs, a_core_s_mnS).reshape({l_, r_, m_, n_, S_});

  // B[l, r, m, n, L, R] = sum_S C[l, r, m, n, S] * phi_right[L, S, R]
  torch::Tensor C_lrmn_S = C.reshape({l_ * r_ * m_ * n_, S_});
  torch::Tensor phi_right_S_LR =
    phi_right_raw_.permute({1, 0, 2}).contiguous().reshape({S_, L_ * R_});
  torch::Tensor B =
    torch::matmul(C_lrmn_S, phi_right_S_LR).reshape({l_, r_, m_, n_, L_, R_});

  // -> [l, m, L, r, n, R] -> [(l*m*L), (r*n*R)]
  B = B.permute({0, 2, 4, 1, 3, 5}).contiguous();
  torch::Tensor dense = B.reshape({l_ * m_ * L_, r_ * n_ * R_});
  if (regularization_ != 0.0) {
    dense = dense + regularization_ * torch::eye(l_ * m_ * L_, dense.options());
  }
  return dense;
}

} // namespace ttnte::linalg::amen
