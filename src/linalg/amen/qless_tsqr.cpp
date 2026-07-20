#include "ttnte/linalg/amen/qless_tsqr.hpp"
#include <ATen/Parallel.h>
#include <cmath>

namespace ttnte::linalg::amen {

namespace {

/// @brief Computes the upper-triangular `R` for a single row-block via
/// Cholesky-QR (`R^T R = Mb^T Mb`), falling back to a discarded-Q
/// `at::linalg_qr` if the Gram matrix isn't numerically SPD. Cholesky-QR is
/// only valid for tall-or-square blocks (`rows >= cols`) -- a wide block's
/// Gram matrix is rank-deficient by construction (rank <= rows < cols),
/// never positive definite, so those go straight to `at::linalg_qr` (which
/// correctly returns a non-square `[rows, cols]` R in that case).
torch::Tensor cholesky_qr_block(const torch::Tensor& Mb)
{
  if (Mb.size(0) >= Mb.size(1)) {
    torch::Tensor gram = torch::matmul(Mb.transpose(0, 1), Mb);
    try {
      torch::Tensor L = torch::linalg_cholesky(gram);
      return L.transpose(0, 1).contiguous();
    } catch (const c10::Error&) {
      // Fall through to the QR fallback below.
    }
  }
  auto [q, r] = torch::linalg_qr(Mb, "reduced");
  return r;
}

} // namespace

torch::Tensor tsqr_r(const torch::Tensor& M, int64_t block_size)
{
  TORCH_CHECK(M.dim() == 2, "tsqr_r expects a 2-D tensor");
  const int64_t rows = M.size(0);
  const int64_t cols = M.size(1);

  if (cols == 0) {
    return torch::empty({0, 0}, M.options());
  }
  if (rows <= cols) {
    // Not tall-skinny: a single Cholesky-QR (or QR fallback) suffices.
    return cholesky_qr_block(M);
  }

  const int64_t bs = std::max(block_size, cols);

  // Leaf level: split into row-blocks and reduce each independently.
  std::vector<torch::Tensor> blocks;
  for (int64_t start = 0; start < rows; start += bs) {
    const int64_t len = std::min(bs, rows - start);
    blocks.push_back(M.narrow(0, start, len));
  }

  std::vector<torch::Tensor> r_factors(blocks.size());
  at::parallel_for(0, blocks.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      r_factors[i] = cholesky_qr_block(blocks[i]);
    }
  });

  // Tree-reduce pairs of R factors: stack [R_i; R_j] (2*cols x cols) and
  // Cholesky-QR again, until a single R remains.
  while (r_factors.size() > 1) {
    std::vector<torch::Tensor> next;
    next.reserve((r_factors.size() + 1) / 2);
    for (size_t i = 0; i < r_factors.size(); i += 2) {
      if (i + 1 < r_factors.size()) {
        torch::Tensor stacked = torch::cat({r_factors[i], r_factors[i + 1]}, 0);
        next.push_back(cholesky_qr_block(stacked));
      } else {
        next.push_back(r_factors[i]);
      }
    }
    r_factors = std::move(next);
  }

  return r_factors.front();
}

std::pair<torch::Tensor, torch::Tensor> qless_orthogonalize(
  const torch::Tensor& M, int64_t block_size)
{
  // The "recover Q via R^{-1}" trick requires a square R, which only exists
  // when M is tall-or-square. For a wide M, R is a non-square [rows, cols]
  // trapezoidal factor and there's no TSQR reduction to do anyway (a single
  // QR already is the cheapest path), so fall back directly.
  if (M.size(0) < M.size(1)) {
    auto [q, r] = torch::linalg_qr(M, "reduced");
    return {q, r};
  }

  torch::Tensor R = tsqr_r(M, block_size);

  // The "recover Q via R^{-1}" trick is only numerically safe when R is
  // well-conditioned. Cholesky-QR can "succeed" (the Gram matrix is
  // technically SPD) while still producing a near-singular R whenever the
  // input is merely ill-conditioned rather than exactly rank-deficient --
  // the triangular solve below then amplifies that to Inf/NaN. Detect this
  // via R's diagonal ratio (a standard, cheap proxy for a triangular
  // matrix's condition number) and fall back to a direct, unconditionally
  // stable Householder QR when it's too large.
  torch::Tensor diag_abs = torch::diagonal(R).abs();
  double max_diag = diag_abs.numel() > 0 ? diag_abs.max().item<double>() : 0.0;
  double min_diag = diag_abs.numel() > 0 ? diag_abs.min().item<double>() : 0.0;
  bool well_conditioned =
    max_diag > 0.0 && std::isfinite(min_diag) && min_diag >= 1e-10 * max_diag;
  if (!well_conditioned) {
    auto [q, r] = torch::linalg_qr(M, "reduced");
    return {q, r};
  }

  torch::Tensor Q =
    torch::linalg_solve_triangular(R, M, /*upper=*/true, /*left=*/false);
  return {Q, R};
}

} // namespace ttnte::linalg::amen
