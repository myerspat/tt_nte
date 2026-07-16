#pragma once

#include "ttnte/solvers/memory_policy.hpp"

namespace ttnte::solvers {

#ifdef USE_CUDA
inline constexpr bool DEFAULT_USE_GPU = true;
inline constexpr MemoryPolicy DEFAULT_MEMORY_POLICY = MemoryPolicy::RESIDENT;
#else
inline constexpr bool DEFAULT_USE_GPU = false;
inline constexpr MemoryPolicy DEFAULT_MEMORY_POLICY = MemoryPolicy::OUT_OF_CORE;
#endif

enum class ExecMode : uint8_t { SYNC, ASYNC };
enum class CommMode : uint8_t { SYNC, ASYNC };

/// @brief Top-level configuration for the domain decomposition solver. Owned
/// by DDStrategy so that a single strategy object carries all tuning knobs.
///
/// A single DAG iteration covers one complete block-Jacobi sweep. For
/// k-eigenvalue problems convergence requires both the relative flux change
/// and the k change to fall below their respective tolerances.
struct DDSolverConfig {
  /// Relative flux-change convergence tolerance (per-patch Frobenius norm).
  double tol = 1e-8;
  /// Maximum number of block-Jacobi iterations.
  int max_iter = 100;
  /// Mode for compute heavy tasks.
  ExecMode exec_mode = ExecMode::ASYNC;
  /// Mode for communication heavy tasks.
  CommMode comm_mode = CommMode::ASYNC;
  /// Number of threads for the TaskScheduler thread pool.
  int num_threads = 4;
  /// Number of CUDA streams for GPU workloads.
  int num_streams = 16;
  /// Use the GPU in compute tasks.
  bool use_gpu = DEFAULT_USE_GPU;
  /// Memory policy for memory management on GPUs.
  MemoryPolicy memory_policy = DEFAULT_MEMORY_POLICY;
  /// Whether to print to terminal.
  bool verbose = false;
  /// Forcing coefficient for the block-Jacobi inner Schwarz tolerance.
  /// Each step() call snapshots its break tolerance as max(tol, tol_forcing *
  /// min_error_), using min_error_ as it stood at the end of the PREVIOUS
  /// step() call. Larger values exit the inner Schwarz earlier (less work per
  /// outer iteration).
  double tol_forcing = 0.1;

  // =================================================================
  // Constructors
  /// @brief Flat constructor — all parameters supplied directly.
  DDSolverConfig(double tol = 1e-8, int max_iter = 100,
    ExecMode exec_mode = ExecMode::ASYNC, CommMode comm_mode = CommMode::ASYNC,
    int num_threads = 4, int num_streams = 16, bool use_gpu = DEFAULT_USE_GPU,
    MemoryPolicy memory_policy = DEFAULT_MEMORY_POLICY,
    double tol_forcing = 0.1, bool verbose = false)
    : tol(tol), max_iter(max_iter), exec_mode(exec_mode), comm_mode(comm_mode),
      num_threads(num_threads), num_streams(num_streams), use_gpu(use_gpu),
      memory_policy(memory_policy), tol_forcing(tol_forcing), verbose(verbose)
  {}
};

} // namespace ttnte::solvers
