import numpy as np
import scipy.sparse as sp
from scikit_tt import TT

from tt_nte.utils.utils import get_degree


class TensorTrain(object):
    def __init__(self, elements, fmt="tt", rank_grp_size=500, threshold=1e-15):
        """
        Construct tensor train from a single numpy array, a list of cores
        as numpy arrays, or a list of list of cores as numpy arrays.
        """
        assert fmt == "tt" or fmt == "qtt"
        self._fmt = fmt
        self._rank_grp_size = rank_grp_size
        self._threshold = threshold

        if isinstance(elements, list):
            # Create scikit_tt train
            self._train = self._construct_scikit_tt(elements)
        elif isinstance(elements, TT):
            self._train = elements

        else:
            raise TypeError(
                f"Unsupported type {type(elements).__name__} given to {type(self).__name__}"
            )

    # =======================================================================
    # Methods

    def _construct_scikit_tt(self, trains):
        """
        Construct TT using scikit_tt.TT objects.
        """
        if isinstance(trains, list) and isinstance(trains[0], list):
            ranks = (
                np.ones(int(len(trains) / self._rank_grp_size), dtype=int)
                * self._rank_grp_size
            )
            if len(trains) % self._rank_grp_size != 0:
                ranks = np.append(ranks, int(len(trains) % self._rank_grp_size))

            tt = TensorTrain(trains[0])
            if self._fmt == "qtt":
                tt.tt2qtt()
            ranks[0] -= 1
            r = 1

            trains.append([])
            trains = np.array(trains, dtype=object)
            trains = trains[:-1]

            for rank in ranks:
                cores = (
                    [
                        np.zeros(
                            (
                                1,
                                trains[r][0].shape[0],
                                trains[r][0].shape[1],
                                rank,
                            )
                        )
                    ]
                    + [
                        np.zeros(
                            (
                                rank,
                                trains[r][i].shape[0],
                                trains[r][i].shape[0],
                                rank,
                            )
                        )
                        for i in range(1, len(trains[r]) - 1)
                    ]
                    + [
                        np.zeros(
                            (
                                rank,
                                trains[r][-1].shape[0],
                                trains[r][-1].shape[1],
                                1,
                            )
                        )
                    ]
                )

                def append_train(tr, idx):
                    # Append first core
                    assert cores[0].shape[1:-1] == tr[0].shape
                    cores[0][0, ..., idx] = tr[0]

                    # Append inner cores
                    for i in range(1, len(cores) - 1):
                        assert cores[i].shape[1:-1] == tr[i].shape
                        cores[i][idx, :, :, idx] = tr[i]

                    # Append final core
                    assert cores[-1].shape[1:-1] == tr[-1].shape
                    cores[-1][idx, ..., 0] = tr[-1]

                sub_trains = trains[r : r + rank]
                vappend_train = np.vectorize(append_train)
                vappend_train(sub_trains, np.arange(rank))
                r += rank

                # Add tensor trains
                add_tt = TensorTrain(cores)
                if self._fmt == "qtt":
                    add_tt.tt2qtt()

                tt += add_tt
                del add_tt

            return tt.train()

        elif isinstance(trains, list) and isinstance(trains[0], np.ndarray):
            for i in range(len(trains)):
                if len(trains[i].shape) == 2:
                    trains[i] = trains[i][np.newaxis, :, :, np.newaxis]
                elif len(trains[i].shape) == 3:
                    trains[i] = trains[i][:, :, np.newaxis, :]
                elif len(trains[i].shape) != 4:
                    raise RuntimeError(
                        f"Shape of core {trains[i].shape} is not supported"
                    )

            return TT(trains)

        else:
            return TT(trains)

    def tt2qtt(self, threshold=1e-15, cores=None):
        """
        Transform TT formatted operator to QTT format.
        """
        # Reshape and permute dims to list of [2] * l_k
        new_dims = []
        if cores is None:
            for dim_size in self._train.row_dims:
                new_dims.append([2] * get_degree(dim_size))

        else:
            for i in range(len(self._train.row_dims)):
                if i in cores:
                    new_dims.append([2] * get_degree(self._train.row_dims[i]))
                else:
                    new_dims.append([self._train.row_dims[i]])

        # Transform operator into QTT format with threshold
        # for SVD decomposition
        self._train = self._train.tt2qtt(new_dims, new_dims, threshold=threshold)

    def norm(self, p=2):
        """
        Calculate the Manhattan norm (p=1) or Euclidean norm (p=2) of a TT.
        """
        return self._train.norm(p)

    def ortho(self, threshold=0.0, max_rank=np.infty):
        self._train.ortho(threshold=threshold, max_rank=max_rank)

    @staticmethod
    def rand(row_dims, col_dims, ranks):
        """
        Create tensor train with random values in cores.
        """
        if not isinstance(ranks, list):
            ranks = [1] + [ranks] * (len(row_dims) - 1) + [1]

        # Define random TT cores
        cores = [
            np.random.rand(ranks[i], row_dims[i], col_dims[i], ranks[i + 1])
            for i in range(len(row_dims))
        ]

        # Create tensor train
        train = TensorTrain(cores)

        return train / train.norm(2)

    @staticmethod
    def zeros(row_dims, col_dims, ranks=1):
        if not isinstance(ranks, list):
            ranks = [1] + [ranks] * (len(row_dims) - 1) + [1]

        # Define random TT cores
        cores = [
            np.zeros((ranks[i], row_dims[i], col_dims[i], ranks[i + 1]))
            for i in range(len(row_dims))
        ]

        # Create tensor train
        train = TensorTrain(cores)

        return train

    # =======================================================================
    # Overloads

    def __add__(self, other):
        if isinstance(other, TensorTrain):
            other = other._train

        train = self._train + other
        train.ortho(threshold=1e-10)
        return TensorTrain(train)

    def __sub__(self, other):
        if isinstance(other, TensorTrain):
            other = other._train

        train = self._train - other
        train.ortho(threshold=1e-10)
        return TensorTrain(train)

    def __mul__(self, other):
        assert not isinstance(other, TensorTrain)

        train = self._train * other
        train.ortho(threshold=1e-10)
        return TensorTrain(train)

    def __truediv__(self, other):
        assert not isinstance(other, TensorTrain)

        train = self._train * (1 / other)
        train.ortho(threshold=1e-10)
        return TensorTrain(train)

    def __matmul__(self, other):
        assert isinstance(other, TensorTrain)
        train = TensorTrain(self._train.dot(other.train()))
        train.ortho(threshold=1e-10)
        return train

    def __repr__(self):
        return self._train.__repr__()

    # =======================================================================
    # Getters

    def train(self, tt_driver="scikit_tt"):
        """
        Get scikit_tt or ttpy TT.
        """
        if tt_driver == "scikit_tt":
            return self._train

        elif tt_driver == "ttpy":
            import tt as ttpy

            if self._train.col_dims == self._train.row_dims:
                return ttpy.matrix.from_list(self.cores)
            else:
                return ttpy.vector.from_list(self.cores)

        else:
            raise RuntimeError(
                "Unsupported tensor train driver"
                + " (only scikit_tt and ttpy are valid)"
            )

    def matricize(self, tt_driver="scikit_tt"):
        """
        Get matricized TT, if 2D then the return type is a
        scipy.sparse.csc_matrix and a numpy array if 1D.
        """
        train = self.train(tt_driver)
        array = None
        if tt_driver == "scikit_tt":
            array = train.matricize()
        else:
            array = train.full()

        if len(array.shape) == 2 and array.shape[0] == array.shape[1]:
            return sp.csc_matrix(array)
        else:
            return array.flatten()

    @property
    def cores(self):
        return self._train.cores

    @property
    def ranks(self):
        return self._train.ranks

    @property
    def row_dims(self):
        return self._train.row_dims

    @property
    def col_dims(self):
        return self._train.col_dims

    @property
    def order(self):
        return self._train.order
