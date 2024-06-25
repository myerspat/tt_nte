# tt_nte

Tensor Trains (TT) applied to the Neutron Transport Equation (NTE).

## Requirements

- scikit_tt
- ttpy
- numpy
- scipy
- pandas

## Installation

Clone the repository:
 
```shell
git clone git@github.com:myerspat/tt_nte.git
```

In `tt_nte/`, run:

```shell
pip install .
```

## Methods

- `tt_nte.methods.DiscreteOrdinates`: Discrete ordiantes approximation applied to the NTE with finite differencing.

## Solvers

- `tt_nte.solvers.Matrix`: Matrix eigenvalue problem solvers using both a generalized eigenvalue solver and power iteration.
- `tt_nte.solvers.ALS`: TT solver using power iteration with Alternating Linear Scheme (ALS).
- `tt_nte.solvers.MALS`: TT solver using power iteration with Modified ALS (MALS) to allow variable rank in the solution.
- `tt_nte.solvers.AMEn`: TT solver using power iteration with the Alternating Minimal Energy Method (AMEn).
- `tt_nte.solvers.GMRES`: TT solver using power iteration with GMRES.
