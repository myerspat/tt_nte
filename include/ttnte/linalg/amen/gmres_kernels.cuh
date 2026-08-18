#pragma once

#include <torch/extension.h>

namespace ttnte::linalg::amen::cuda {

/// @brief Applies one GMRES Arnoldi iteration's sequential Givens-rotation
/// update entirely on-device, replacing a host-side loop that would
/// otherwise need O(k) `.item<T>()` round trips (one per already-known
/// rotation): the update only needs O(1) extra scalar state -- a "carry"
/// value threaded through the `k` previous rotations -- so it runs as a
/// single-thread kernel. The goal isn't to parallelize the (inherently
/// sequential) arithmetic, it's to avoid the host round trips; see
/// src/linalg/amen/gmres_kernels.cu for the derivation. Generic to any
/// caller of `gmres_solve_cpu`/`gmres_solve_gpu` (see gmres_local.hpp) --
/// not specific to the AMEn sweep.
/// @param h Raw (pre-rotation) Gram-Schmidt coefficients, shape `[k+1]`.
/// @param vnorm The newly-orthogonalized Krylov vector's norm (0-dim), i.e.
/// the un-rotated `(k+2)`-th Hessenberg entry.
/// @param cs,sn Rotation-coefficient buffers, shape `[m]`; index `k` is
/// written, indices `[0,k)` are read.
/// @param beta Right-hand-side coefficients, shape `[m+1]`; indices `k` and
/// `k+1` are read/written in place.
/// @param k The current Arnoldi iteration index.
/// @return The rotated Hessenberg column, shape `[k+1]` (store as `R`'s
/// `k`-th column).
torch::Tensor fused_givens_apply(const torch::Tensor& h,
  const torch::Tensor& vnorm, torch::Tensor& cs, torch::Tensor& sn,
  torch::Tensor& beta, int64_t k);

} // namespace ttnte::linalg::amen::cuda
