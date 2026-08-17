#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/linalg/tt_config.hpp"
#include "ttnte/parallel/boundary_communicator.hpp"
#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/solvers/solver_configs.hpp"
#include "ttnte/task/task_graph.hpp"
#include <memory>
#include <unordered_map>

namespace ttnte::solvers {

/// @brief Strategy class for the domain decomposition solver.
class DDStrategy {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<DDStrategy>;
  using SystemPtr = std::shared_ptr<linalg::LinearSystem>;

protected:
  // =================================================================
  // Protected data
  /// A shared pointer to the local system solver.
  LocalSolver::Ptr local_solver_;
  /// Solver configuration (convergence tolerances, rounding, etc.).
  DDSolverConfig config_;
  /// Dynamic low-rank tensor network configuration.
  std::shared_ptr<linalg::TTConfig> tt_config_;
  /// Minimum error reached thus far. A pure running minimum -- DDSolver::
  /// step() is responsible for turning this into an actual Schwarz break
  /// tolerance (snapshotted once per step() call; see get_min_error()'s doc).
  double min_error_ = 1.0;

  // =================================================================
  // Protected constructors
  DDStrategy(DDSolverConfig config = {});

public:
  virtual ~DDStrategy() = default;

  // =================================================================
  // Public methods
  /// @brief Build the iteration DAG for this strategy (CPU path).
  /// Called by DDSolver::build_iteration_dag when use_gpu() is false.
  /// @param dag             Task graph to populate.
  /// @param local_systems   Systems local to this MPI rank.
  /// @param gid_to_local    GID → local_systems index map.
  /// @param boundary_comms  Per-face MPI communicators.
  /// @param boundary_cfg    Rounding config for boundary communication tasks.
  virtual void build_cpu_iteration_dag(task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems,
    const std::unordered_map<int64_t, size_t>& gid_to_local,
    const parallel::BoundaryCommunicator& boundary_comms) const;

  /// @brief Build the iteration DAG for this strategy (GPU path).
  /// Called by DDSolver::build_iteration_dag when use_gpu() is true. GPU
  /// tasks in the built DAG get their stream from whichever TaskScheduler
  /// worker thread executes them (see parallel::StreamPool::current_stream())
  /// -- no stream pool is threaded through DAG construction itself.
  /// @param dag             Task graph to populate.
  /// @param local_systems   Systems local to this MPI rank.
  /// @param gid_to_local    GID → local_systems index map.
  /// @param boundary_comms  Per-face MPI communicators.
  virtual void build_gpu_iteration_dag(task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems,
    const std::unordered_map<int64_t, size_t>& gid_to_local,
    const parallel::BoundaryCommunicator& boundary_comms) const;

  /// @brief Track the best (smallest) partial-current error observed so far
  /// and forward it (plus `rank_metric`) to the local solver's own forcing.
  /// Deliberately does NOT compute a Schwarz break tolerance here -- doing
  /// that on every call (i.e. every inner iteration) would make the
  /// tolerance chase the very error it's being compared against (tol ==
  /// tol_forcing * this iteration's own error), so `error < tol` could only
  /// ever succeed once tol_forcing * min_error_ drops below the hard floor
  /// config_.tol -- see DDSolver::step(), which snapshots the break
  /// tolerance ONCE per step() call instead, from min_error_ as it stood at
  /// the end of the PREVIOUS call.
  /// @param rank_metric See `Solver::update_convergence_criteria`'s doc --
  /// forwarded as-is to the local solver.
  void update_convergence_criteria(double error, double rank_metric = 0.0)
  {
    if (error < min_error_ && error > 0) {
      min_error_ = error;
    }

    local_solver_->update_convergence_criteria(error, rank_metric);
    tt_config_->eps = local_solver_->get_eps();
    tt_config_->max_rank = local_solver_->get_max_rank();
  }

  // =================================================================
  // Public getters / setters
  /// @return The solver configuration (convergence tolerances, rounding, etc.).
  const DDSolverConfig& get_config() const noexcept { return config_; }
  /// @param config The new solver configuration.
  void set_config(const DDSolverConfig& config) { config_ = config; }
  /// @return The current local linear solver.
  const LocalSolver::Ptr& get_local_solver() const noexcept
  {
    return local_solver_;
  }
  /// @param local_solver The new local solver for this strategy.
  void set_local_solver(const LocalSolver::Ptr& local_solver)
  {
    local_solver_ = local_solver;
    tt_config_->eps = local_solver_->get_eps();
    tt_config_->max_rank = local_solver_->get_max_rank();
  }

  /// @return The best (smallest) partial-current error observed across every
  /// step() call so far -- a pure running minimum, not itself a tolerance.
  /// DDSolver::step() derives its Schwarz break tolerance from this,
  /// snapshotted once per call.
  double get_min_error() const noexcept { return min_error_; }
};

} // namespace ttnte::solvers
