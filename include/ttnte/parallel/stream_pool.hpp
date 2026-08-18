#pragma once

#include "ttnte/parallel/stream_handle.hpp"
#include <c10/util/SmallVector.h>
#include <memory>
#include <optional>
#include <torch/extension.h>

namespace ttnte::parallel {

/// @brief A stream pool manager for torch CUDA streams.
class StreamPool {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<StreamPool>;

private:
  // =================================================================
  // Private data
  /// Vector of available streams.
  c10::SmallVector<StreamHandle, 16> streams_;
  /// Total number of streams this pool was created with -- fixed at
  /// construction, unlike streams_ (which shrinks as streams are claimed).
  size_t total_streams_ = 0;
  /// GPU device for the streams.
  torch::Device device_;
  /// Mutex of the class for thread safety.
  std::mutex mutex_;

protected:
  // =================================================================
  // Protected constructors
  StreamPool(int num_streams = 16);

public:
  ~StreamPool() = default;

  // Prevent copying
  StreamPool(const StreamPool&) = delete;
  StreamPool& operator=(const StreamPool&) = delete;

  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new instance of the stream pool.
  /// Each owner (e.g. one per TaskScheduler) should create its own pool
  /// rather than sharing one process-wide -- streams claimed via
  /// claim_for_this_thread() are held for the claiming thread's whole
  /// lifetime and never returned, so a pool shared across independently
  /// created/destroyed ThreadPools would eventually be permanently
  /// exhausted by threads that have long since exited.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new StreamPool(std::forward<Args>(args)...));
  }
  /// @brief Try to acquire a free stream. If the optional pointer is empty then
  /// there is not an available stream.
  /// @return An optional pointer to an available stream.
  std::optional<StreamHandle> try_acquire();
  /// @brief Return the stream to the stream pool.
  /// @param stream The returning stream.
  void release(const StreamHandle& stream);
  /// @brief Claim one stream from this pool for the calling thread, held for
  /// the remainder of the thread's lifetime (never released back to the
  /// pool) -- gives every thread a fixed, exclusively-owned stream instead of
  /// threads round-robining through a shared pool, which would otherwise let
  /// a single thread's per-thread cuBLAS/cuSOLVER handle end up bound to
  /// multiple concurrently-in-flight streams (not safe -- see
  /// TaskScheduler's worker init, the only intended caller of this). No-op
  /// if the pool is empty (CPU-only build/run, or CUDA unavailable at
  /// runtime -- see the constructor). Must be called at most once per thread.
  void claim_for_this_thread();
  /// @return The calling thread's own stream, previously claimed via
  /// claim_for_this_thread().
  /// @throws ttnte::utils::runtime_error if this thread never claimed one.
  static const StreamHandle& current_stream();

  // =================================================================
  // Public getters
  /// @return The device for these streams.
  torch::Device get_device() const noexcept { return device_; }
  /// @return The total number of streams in this pool (available or
  /// claimed) -- 0 if CUDA is unavailable at runtime, regardless of the
  /// `num_streams` requested at construction (see the constructor).
  size_t size() const noexcept;
};

} // namespace ttnte::parallel
