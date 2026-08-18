#pragma once

#include <condition_variable>
#include <functional>
#include <latch>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ttnte::parallel {

/// @brief A thread pool manager.
class ThreadPool {
private:
  // =================================================================
  // Private data
  /// Vector of worker threads.
  std::vector<std::thread> workers_;
  /// A queue of tasks to execute among the workers.
  std::queue<std::function<void()>> tasks_;

  /// Mutex for thread safety.
  std::mutex queue_mutex_;
  /// Condition variable to avoid spinning CPU cycles when there are no workers
  /// available.
  std::condition_variable cv_;
  /// Boolean to terminate the thread pool.
  bool stop_;

  /// Counts down to zero once every worker thread has finished its one-time
  /// startup work (CUDA device init, etc.).
  std::latch init_latch_;

public:
  // =================================================================
  // Public constructors
  /// @param num_threads Number of worker threads to spawn.
  /// @param init_fn Optional function run once on each worker thread, before
  /// it enters its work loop. ThreadPool itself has no idea what this does
  /// (e.g. CUDA device pinning, stream claiming) -- that policy belongs to
  /// the caller (see TaskScheduler), keeping this class fully generic.
  explicit ThreadPool(
    size_t num_threads = 4, std::function<void()> init_fn = {});
  ~ThreadPool();

  // Prevent copying which would wreck the thread management
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // =================================================================
  // Public methods
  /// @brief Block until every worker thread has completed its init_fn.
  void wait_for_init() { init_latch_.wait(); }

  /// @brief Push a new task to the queue.
  /// @param f The function that needs to be executed.
  template<class FuncType>
  void push_task(FuncType&& f)
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("Cannot push tasks into a stopped ThreadPool");
      }
      tasks_.emplace(std::forward<FuncType>(f));
    }
    // Wake up exactly one sleeping worker to handle this new task
    cv_.notify_one();
  }
};

} // namespace ttnte::parallel
