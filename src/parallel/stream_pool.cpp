#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/utils/exception.hpp"
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/cuda.h>

namespace {
// Each thread's own permanently-claimed stream, set at most once via
// StreamPool::claim_for_this_thread(). Namespace-scoped (not a StreamPool
// member) since ownership is a property of the thread, not of any single
// StreamPool instance.
thread_local std::optional<ttnte::parallel::StreamHandle> t_claimed_stream;
} // namespace

namespace ttnte::parallel {

// =================================================================
// Protected constructors
StreamPool::StreamPool(int num_streams)
  : device_(parallel::ParallelContext::instance().device())
{
  if (torch::cuda::is_available()) {
    // Retrieve the number of streams requested
    auto* guard_impl = c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA);

    streams_.reserve(num_streams);
    for (int i = 0; i < num_streams; i++) {
      streams_.emplace_back(
        guard_impl->getStreamFromGlobalPool(device_, false));
    }
    total_streams_ = static_cast<size_t>(num_streams);
  }
}

// =================================================================
// Public methods
std::optional<StreamHandle> StreamPool::try_acquire()
{
  std::unique_lock<std::mutex> lock(mutex_);

  if (streams_.empty()) {
    return std::nullopt;
  }

  auto stream = streams_.back();
  streams_.pop_back();
  return stream;
}

void StreamPool::release(const StreamHandle& stream)
{
  std::unique_lock<std::mutex> lock(mutex_);
  streams_.push_back(std::move(stream));
}

void StreamPool::claim_for_this_thread()
{
  if (size() == 0) {
    // CPU-only build/run, or CUDA unavailable at runtime -- nothing to
    // claim.
    return;
  }

  auto stream = try_acquire();
  if (!stream.has_value()) {
    throw utils::runtime_error(
      "ttnte::parallel::StreamPool::claim_for_this_thread",
      "No stream available to claim, the pool must contain exactly one "
      "stream per thread that will call this");
  }
  t_claimed_stream = stream;
}

const StreamHandle& StreamPool::current_stream()
{
  if (!t_claimed_stream.has_value()) {
    throw utils::runtime_error("ttnte::parallel::StreamPool::current_stream",
      "This thread never claimed a stream via claim_for_this_thread()");
  }
  return *t_claimed_stream;
}

size_t StreamPool::size() const noexcept
{
  return total_streams_;
}

} // namespace ttnte::parallel
