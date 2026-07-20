#pragma once

// Q-less, blocked Tall-Skinny QR (TSQR).
//
// Algorithmic background: Roehrig-Zoellner, Becklas, Thies, Basermann (2025),
// "Performance of linear solvers in tensor-train format on current multicore
// architectures", Sec. 5.1, itself based on the communication-avoiding TSQR
// of Demmel, Grigori, Hoemmen, Langou (2012), "Communication-optimal
// Parallel and Sequential QR and LU Factorizations", SIAM J. Sci. Comput.
// The `pitts` library (Roehrig-Zoellner et al., BSD-3-Clause, DLR) implements
// the same idea as `block_TSQR` in `pitts_multivector_tsqr_impl.hpp`; this is
// an independent reimplementation against ttnte's own tensor types (no code
// copied), using Cholesky-QR instead of block-Householder reflections for
// the leaf reduction, which maps onto a single GEMM + a small Cholesky
// factorization per block -- a better fit for cuBLAS/cuSOLVER than the
// inherently sequential Householder panel updates LAPACK/cuSOLVER's own
// `geqrf` uses for tall-skinny inputs.

#include <torch/extension.h>
#include <utility>

namespace ttnte::linalg::amen {

/// @brief Computes only the upper-triangular factor `R` of the (thin) QR
/// decomposition `M = Q R`, without ever forming `Q`. Splits `M`'s rows into
/// blocks of `block_size`, computes each block's local `R` via Cholesky-QR
/// (`R_k^T R_k = M_k^T M_k`), then tree-reduces pairs of local `R` factors
/// the same way until a single `R` remains. Falls back to a plain
/// `at::linalg_qr` (discarding `Q`) for any block whose Gram matrix is not
/// numerically SPD, for robustness on ill-conditioned inputs.
/// @param M A tall-skinny matrix, `rows >= cols`.
/// @param block_size Row-block size for the reduction; must be `>= cols`.
/// @return The upper-triangular `R` factor, shape `[cols, cols]`.
torch::Tensor tsqr_r(const torch::Tensor& M, int64_t block_size);

/// @brief Q-less orthogonalization of a tall-skinny matrix. Computes
/// `R = tsqr_r(M)` and recovers the orthonormalized factor as
/// `Q = M @ R^{-1}` (one triangular solve) instead of accumulating
/// Householder reflectors, following eq. 23 of Roehrig-Zoellner et al. 2025.
/// @param M A tall-skinny matrix, `rows >= cols`.
/// @param block_size Row-block size for the underlying `tsqr_r` reduction.
/// @return `{Q, R}` with `Q` orthonormal-column (`Q^T Q ~= I`) and
/// `M ~= Q @ R`.
std::pair<torch::Tensor, torch::Tensor> qless_orthogonalize(
  const torch::Tensor& M, int64_t block_size);

} // namespace ttnte::linalg::amen
