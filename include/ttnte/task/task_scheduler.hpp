#pragma once

#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/parallel/thread_pool.hpp"
#include "ttnte/task/task_graph.hpp"
#include "ttnte/utils/label.hpp"
#include <mutex>

namespace ttnte::task {

/// @brief This class executes the graph held by the TaskGraph.
class TaskScheduler {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<TaskScheduler>;

private:
  // =================================================================
  // Private data
  /// Label of the scheduler.
  Label label_;
  /// The GPU stream pool -- own instance (not process-wide), sized to
  /// exactly `num_threads`: each worker thread of thread_pool_ permanently
  /// claims one stream (see StreamPool::claim_for_this_thread()) and never
  /// releases it, so a pool shared across independently created/destroyed
  /// TaskSchedulers would eventually be exhausted by threads that have long
  /// since exited. Declared before thread_pool_ so it exists before
  /// thread_pool_'s workers try to claim from it. Empty (size() == 0) on a
  /// CPU-only build/run -- see StreamPool's constructor.
  parallel::StreamPool::Ptr stream_pool_;
  /// The thread pool of the scheduler.
  parallel::ThreadPool thread_pool_;

  /// The mutex for stopping race conditions.
  std::mutex mutex;

public:
  // =================================================================
  // Public constructor
  /// @brief Constructor for the scheduler. Builds its own GPU stream pool
  /// sized to exactly `num_threads` and pins each worker thread to the GPU
  /// device, having it permanently claim one of the pool's streams (see
  /// StreamPool::claim_for_this_thread()).
  /// @param num_threads The number of threads for the thread pool.
  /// @param label The label of the scheduler.
  TaskScheduler(
    size_t num_threads = 4, std::optional<std::string> label = std::nullopt);

  // =================================================================
  // Public methods
  /// @brief Block until every worker thread has completed its init_fn.
  void wait_for_init() { thread_pool_.wait_for_init(); }

  /// @brief Execute a task graph.
  /// @param The TaskGraph to execute.
  void execute(TaskGraph& graph);

  // =================================================================
  // Public getters / setters
  /// @return The label of the scheduler.
  const Label& get_label() const noexcept { return label_; }
  /// @return The thread pool of the scheduler.
  const parallel::ThreadPool& get_thread_pool() const noexcept
  {
    return thread_pool_;
  }
  /// @return The GPU stream pool.
  const parallel::StreamPool::Ptr& get_stream_pool() const noexcept
  {
    return stream_pool_;
  }
};

} // namespace ttnte::task
