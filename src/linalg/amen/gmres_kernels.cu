// Fused CUDA kernels for the native GMRES local solve (see gmres_local.hpp).
// Reusable by any caller of gmres_solve_gpu(), not specific to the AMEn
// sweep -- see include/ttnte/linalg/amen/amen_config.hpp for the AMEn
// algorithmic background this solver is otherwise built for
// (Roehrig-Zoellner et al. 2025).
#include "ttnte/linalg/amen/gmres_kernels.cuh"
#include <ATen/cuda/CUDAContext.h>

namespace ttnte::linalg::amen::cuda {

namespace {

template<typename scalar_t>
__global__ void givens_apply_kernel(const scalar_t* __restrict__ h,
  const scalar_t* __restrict__ vnorm, scalar_t* __restrict__ cs,
  scalar_t* __restrict__ sn, scalar_t* __restrict__ r_col,
  scalar_t* __restrict__ beta, int64_t k)
{
  // Sequentially apply the k already-known rotations to (h[0..k], *vnorm).
  // Rotation i only ever touches indices (i, i+1); index i+1 is untouched by
  // any rotation before i, so it's still the raw h[i+1]. That means the only
  // state that needs to carry from one rotation to the next is the single
  // running value at the "current" index -- no array beyond h/r_col itself
  // is needed, hence a single thread with O(1) extra scalars suffices.
  scalar_t carry = h[0];
  for (int64_t i = 0; i < k; ++i) {
    scalar_t right = h[i + 1];
    scalar_t c_i = cs[i];
    scalar_t s_i = sn[i];
    scalar_t new_left = c_i * carry + s_i * right;
    scalar_t new_right = -s_i * carry + c_i * right;
    r_col[i] = new_left;
    carry = new_right;
  }

  // New rotation zeroing the (k+1)-th entry (*vnorm) against the carry.
  scalar_t a = carry;
  scalar_t bb = *vnorm;
  scalar_t denom = sqrt(a * a + bb * bb);
  scalar_t c = denom > scalar_t(0) ? a / denom : scalar_t(1);
  scalar_t s = denom > scalar_t(0) ? bb / denom : scalar_t(0);
  cs[k] = c;
  sn[k] = s;
  r_col[k] = c * a + s * bb;

  scalar_t beta_k = beta[k];
  beta[k] = c * beta_k;
  beta[k + 1] = -s * beta_k;
}

} // namespace

torch::Tensor fused_givens_apply(const torch::Tensor& h,
  const torch::Tensor& vnorm, torch::Tensor& cs, torch::Tensor& sn,
  torch::Tensor& beta, int64_t k)
{
  TORCH_CHECK(h.is_cuda() && vnorm.is_cuda() && cs.is_cuda() && sn.is_cuda() &&
      beta.is_cuda(),
    "fused_givens_apply expects CUDA tensors");
  TORCH_CHECK(cs.is_contiguous() && sn.is_contiguous() && beta.is_contiguous(),
    "fused_givens_apply expects cs/sn/beta to be contiguous (allocated once "
    "per restart and only ever written at a single index)");

  torch::Tensor h_c = h.contiguous();
  torch::Tensor r_col = torch::empty({k + 1}, h.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "fused_givens_apply", [&] {
    givens_apply_kernel<scalar_t><<<1, 1, 0, stream>>>(h_c.data_ptr<scalar_t>(),
      vnorm.data_ptr<scalar_t>(), cs.data_ptr<scalar_t>(),
      sn.data_ptr<scalar_t>(), r_col.data_ptr<scalar_t>(),
      beta.data_ptr<scalar_t>(), k);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return r_col;
}

} // namespace ttnte::linalg::amen::cuda
