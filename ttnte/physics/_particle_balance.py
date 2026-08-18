"""Optional pandas convenience for the particle-balance diagnostics.

All of the actual computation and cross-rank consolidation happens in C++ --
see `TransportSolution.compute_patch_balances()`/`patch_balance_table()`/
`global_balance()`. This module only attaches a `to_dataframe()` method onto
the bound `PatchBalanceTable`/`GlobalBalance` classes, the same way plotting
is attached to `ttnte.cad.Patch` (see `ttnte/cad/_patch.py`). pandas is a
soft dependency, only imported when `to_dataframe()` is actually called.
"""

from ttnte.cpp.ttnte_python.physics import GlobalBalance, PatchBalanceTable

_SCALAR_FIELDS = (
    "fixed_source",
    "fission_source",
    "absorption",
    "scatter_in",
    "scatter_out",
    "leakage",
)


def _global_balance_to_dataframe(self):
    """Convert a `GlobalBalance` into a pandas DataFrame (rows = quantities, columns =
    groups)."""
    import pandas as pd

    data = {field: getattr(self, field).tolist() for field in _SCALAR_FIELDS}
    data["dd_residual"] = self.dd_residual.tolist()
    return pd.DataFrame(data).transpose()


def _patch_balance_table_to_dataframe(self):
    """Convert a `PatchBalanceTable` into a pandas DataFrame (rows = patch GID, columns
    = quantities, each summed over group)."""
    import pandas as pd

    rows = {
        pb.gid: {field: getattr(pb, field).sum().item() for field in _SCALAR_FIELDS}
        for pb in self.patches
    }
    return pd.DataFrame(rows).transpose()


GlobalBalance.to_dataframe = _global_balance_to_dataframe
PatchBalanceTable.to_dataframe = _patch_balance_table_to_dataframe
