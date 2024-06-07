import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

import utils


def plot_qtt_svd(method, axs=None, figsize=(10, 15)):
    """
    Plot SVD for all cores in each of the three operators when converting
    from TT to QTT format.
    """
    if axs is None:
        plt.clf()
        _, axs = plt.subplots(3, figsize=figsize)

    tts = [method.H("tt"), method.F("tt"), method.S("tt")]
    tt_names = ["LHS Operator", "Fission Operator", "Scattering Operator"]

    for tt_idx in range(len(tts)):
        tt = tts[tt_idx]
        tt_tensor = tt.copy()

        # QTT shaping
        new_dims = []
        for dim_size in tt.row_dims:
            new_dims.append([2] * utils.get_degree(dim_size))

        for i in range(tt.order):
            # Get core features
            core = tt_tensor.cores[i]
            rank = tt_tensor.ranks[i]
            row_dim = tt_tensor.row_dims[i]
            col_dim = tt_tensor.col_dims[i]

            c = plt.cm.rainbow(np.linspace(0, 1, tt.order))[i]

            for j in range(len(new_dims[i]) - 1):
                # Set new row_dim and col_dim for reshape
                row_dim = int(row_dim / new_dims[i][j])
                col_dim = int(col_dim / new_dims[i][j])

                # Reshape and transpose core
                core = core.reshape(
                    rank,
                    new_dims[i][j],
                    row_dim,
                    new_dims[i][j],
                    col_dim,
                    tt_tensor.ranks[i + 1],
                ).transpose([0, 1, 3, 2, 4, 5])

                # Apply SVD to split core
                [_, s, v] = svd(
                    core.reshape(
                        rank * new_dims[i][j] ** 2,
                        row_dim * col_dim * tt_tensor.ranks[i + 1],
                    ),
                    full_matrices=False,
                    overwrite_a=True,
                    check_finite=False,
                    lapack_driver="gesvd",
                )

                # Plot SVD singular values
                if j == 0:
                    axs[tt_idx].plot(s, c=c, label=f"Core {i}")
                else:
                    axs[tt_idx].plot(s, c=c)

                # Update residual core and rank
                core = np.diag(s).dot(v)
                rank = s.shape[0]

        axs[tt_idx].set(xlabel="Singular Value Index", title=tt_names[tt_idx])
        axs[tt_idx].legend()
        axs[tt_idx].set_yscale("log")

    return axs
