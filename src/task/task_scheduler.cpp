#include "ttnte/task/task_scheduler.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include <chrono>
#include <torch/cuda.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAFunctions.h>

namespace {
void pin_cuda_device(torch::DeviceIndex device_idx)
{
  c10::cuda::set_device(device_idx);
}
} // namespace

#else

namespace {
void pin_cuda_device(torch::DeviceIndex device_idx) {}
} // namespace

#endif

namespace {

/// @brief Build the per-worker-thread startup routine: pin the CUDA device
/// (matching what ThreadPool used to do internally) and permanently claim
/// one stream from `stream_pool` for this thread (see
/// StreamPool::claim_for_this_thread() -- the reason every worker needs its
/// own, never-shared stream; a no-op on a CPU-only build/run, since the pool
/// is then empty).
std::function<void()> build_worker_init(
  const ttnte::parallel::StreamPool::Ptr& stream_pool)
{
  auto device_idx =
    ttnte::parallel::ParallelContext::instance().device().index();

  return [device_idx, stream_pool]() {
    if (torch::cuda::is_available()) {
      pin_cuda_device(device_idx);
    }
    stream_pool->claim_for_this_thread();
  };
}

} // namespace

namespace ttnte::task {

TaskScheduler::TaskScheduler(
  size_t num_threads, std::optional<std::string> label)
  : label_(label.has_value() ? Label::from_string(label.value())
                             : Label::create_internal()),
    stream_pool_(parallel::StreamPool::create(num_threads)),
    thread_pool_(num_threads, build_worker_init(stream_pool_))
{}

void TaskScheduler::execute(TaskGraph& graph)
{
  // Lock class from multiple threads trying to execute the same graph
  std::lock_guard<std::mutex> lock(mutex);

  int tasks_completed = 0;
  const int total_tasks = graph.size();

  while (tasks_completed < total_tasks) {
    bool made_progress = false;

    for (auto& task : graph.get_tasks()) {
      TaskStatus current_status = task.get_status();

      // Check dependencies
      if (current_status == TaskStatus::WAITING && task.check_dependencies()) {
        current_status = TaskStatus::READY;
        task.update_status(current_status);
        made_progress = true;
      }

      // Dispatch READY tasks
      if (current_status == TaskStatus::READY) {

        // Transition out of READY on the main thread before dispatching
        current_status = TaskStatus::RUNNING;
        task.update_status(current_status);
        made_progress = true;

        DeviceTarget target = task.get_target();

        if (target == DeviceTarget::CPU_SYNC ||
            target == DeviceTarget::GPU_SYNC ||
            target == DeviceTarget::NETWORK_SYNC) {

          // Execute blocking payload directly on the main scheduler thread
          task.execute();
          assert(task.get_status() == TaskStatus::COMPLETED);

        } else if (target == DeviceTarget::NETWORK_ASYNC) {

          // Execute MPI tasks inline on the main scheduler thread.
          // MPI calls (MPI_Improbe, MPI_Imrecv, MPI_Test) are not thread-safe
          // unless MPI_THREAD_MULTIPLE is available; running them here keeps
          // all MPI on one thread and avoids data races in OpenMPI internals.
          // The operations themselves are non-blocking, so the scheduler
          // continues dispatching and polling other tasks between retries.
          task.execute();

        } else { // CPU_ASYNC, GPU_ASYNC

          // Hand off the initial dispatch to a background worker
          thread_pool_.push_task([&task]() {
            // The payload performs the work or hardware dispatch, and
            // inherently updates its own atomic status to POLLING or COMPLETED
            task.execute();
          });
        }
      }

      // Fetch status again in case an async thread just finished its dispatch
      current_status = task.get_status();

      // Poll hardware for asynchronous tasks
      if (current_status == TaskStatus::POLLING) {
        // The payload now acts as a non-blocking check
        task.execute();

        // Fetch status one last time for the final count check
        current_status = task.get_status();
      }

      // Mark completed tasks
      if (current_status == TaskStatus::COMPLETED && !task.is_counted()) {
        tasks_completed++;
        task.count();
        made_progress = true;
      }
    }

    // If a full pass of the DAG resulted in no dependencies met, no dispatches,
    // and no completed polls, yield the CPU time slice to the OS.
    if (!made_progress) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
}

} // namespace ttnte::task
