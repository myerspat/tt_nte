#include "ttnte/linalg/amen/rank1_preconditioner.hpp"

namespace ttnte::linalg::amen {

Rank1Preconditioner Rank1Preconditioner::build(
  const TTEngine& A, double sv_floor_ratio)
{
  // Truncate A to TT-rank 1: each resulting core has shape [1, m_i, n_i, 1].
  TTEngine A_rank1 = A.round(1e-10, 1);

  Rank1Preconditioner prec;
  const auto& cores = A_rank1.get_cores();
  prec.left_factors_.reserve(cores.size());
  prec.right_factors_.reserve(cores.size());
  prec.right_inverse_factors_.reserve(cores.size());

  for (const auto& core : cores) {
    torch::Tensor mat = core.reshape({core.size(1), core.size(2)});

    auto [U, S, Vh] = torch::linalg_svd(mat, /*full_matrices=*/false);
    torch::Tensor s_floor = S.max() * sv_floor_ratio;
    torch::Tensor s_clamped = torch::clamp_min(S, s_floor);
    torch::Tensor inv_sqrt_s = torch::rsqrt(s_clamped);
    torch::Tensor sqrt_s = torch::sqrt(s_clamped);

    // L_i = S_i^{-1/2} U_i^T, shape [rank_i, m_i].
    prec.left_factors_.push_back(
      (inv_sqrt_s.unsqueeze(1) * U.transpose(0, 1)).contiguous());
    // R_i = V_i S_i^{-1/2} = (Vh^T) * S_i^{-1/2}, shape [n_i, rank_i].
    prec.right_factors_.push_back(
      (Vh.transpose(0, 1) * inv_sqrt_s.unsqueeze(0)).contiguous());
    // R_i^{-1} = S_i^{1/2} V_i^T = S_i^{1/2} Vh, shape [rank_i, n_i] (used to
    // map a user-supplied x0 into the preconditioned/canonical space for
    // warm-starting: y0 = R_i^{-1} x0).
    prec.right_inverse_factors_.push_back(
      (sqrt_s.unsqueeze(1) * Vh).contiguous());
  }

  return prec;
}

std::vector<torch::Tensor> Rank1Preconditioner::apply_left(
  const std::vector<torch::Tensor>& vector_cores) const
{
  std::vector<torch::Tensor> out;
  out.reserve(vector_cores.size());
  for (size_t i = 0; i < vector_cores.size(); ++i) {
    // vector_cores[i]: [r_l, m_i, r_r]; left_factors_[i]: [rank_i, m_i].
    out.push_back(
      torch::einsum("lmr,pm->lpr", {vector_cores[i], left_factors_[i]}));
  }
  return out;
}

std::vector<torch::Tensor> Rank1Preconditioner::apply_right(
  const std::vector<torch::Tensor>& vector_cores) const
{
  std::vector<torch::Tensor> out;
  out.reserve(vector_cores.size());
  for (size_t i = 0; i < vector_cores.size(); ++i) {
    // vector_cores[i]: [r_l, rank_i, r_r]; right_factors_[i]: [n_i, rank_i].
    out.push_back(
      torch::einsum("lpr,np->lnr", {vector_cores[i], right_factors_[i]}));
  }
  return out;
}

std::vector<torch::Tensor> Rank1Preconditioner::apply_right_inverse(
  const std::vector<torch::Tensor>& vector_cores) const
{
  std::vector<torch::Tensor> out;
  out.reserve(vector_cores.size());
  for (size_t i = 0; i < vector_cores.size(); ++i) {
    // vector_cores[i]: [r_l, n_i, r_r]; right_inverse_factors_[i]:
    // [rank_i, n_i].
    out.push_back(torch::einsum(
      "lnr,qn->lqr", {vector_cores[i], right_inverse_factors_[i]}));
  }
  return out;
}

std::vector<torch::Tensor> Rank1Preconditioner::sandwich_operator(
  const std::vector<torch::Tensor>& operator_cores) const
{
  std::vector<torch::Tensor> out;
  out.reserve(operator_cores.size());
  for (size_t i = 0; i < operator_cores.size(); ++i) {
    // operator_cores[i]: [r_l, m_i, n_i, r_r]; L_i: [rank_i, m_i];
    // R_i: [n_i, rank_i].
    out.push_back(torch::einsum("pm,amnc,nq->apqc",
      {left_factors_[i], operator_cores[i], right_factors_[i]}));
  }
  return out;
}

} // namespace ttnte::linalg::amen
