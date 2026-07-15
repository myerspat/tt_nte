#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/solvers/solver.hpp"
#include <memory>
#include <tuple>

namespace ttnte::solvers {

/// @brief The local solver base class.
class LocalSolver : public Solver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<LocalSolver>;

protected:
  // =================================================================
  // Protected data
  Systems local_systems_;
  double min_error_ = 1.0;

public:
  virtual ~LocalSolver() = default;

  // =================================================================
  // Public methods
  /// @brief Solve the local linear system.
  /// @param local_system The local linear system to be solved.
  virtual void solve(const linalg::LinearSystem::Ptr& local_system) = 0;

  /// @brief Compute the right-hand-side of the linear system based on the new
  /// boundary conditions.
  /// @param local_system The local linear system to be solved.
  /// @return A tuple of A, b, x0.
  std::tuple<linalg::Operator, linalg::State, linalg::State> presolve(
    const linalg::LinearSystem::Ptr& sys) const;

  /// @brief Post solve cleanup.
  /// @param local_system The local linear system to be solved.
  void postsolve(
    const linalg::LinearSystem::Ptr& sys, const linalg::State& x) const;

  /// @brief Store the systems to be solved (one per patch).
  void init(const Systems& local_systems) override;
  /// @brief Solve every registered system in turn.
  void step() override;
  /// @brief Track the best (smallest) error observed so far. Concrete
  /// solvers (e.g. AMEnSolver) override this to also derive their own
  /// forcing from min_error_.
  void update_convergence_criteria(double error) override
  {
    if (error < min_error_ && error > 0) {
      min_error_ = error;
    }
  }

  // =================================================================
  // Public getters / setters
  const Systems& get_local_systems() const override { return local_systems_; }
  /// @brief Equivalent to init() -- re-registers the systems to be solved.
  void set_local_systems(const Systems& local_systems) override
  {
    init(local_systems);
  }
};

} // namespace ttnte::solvers
