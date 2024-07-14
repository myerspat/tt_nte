import numpy as np
import scipy.sparse as sp
from scikit_tt import TT

from tt_nte.utils.utils import get_degree


class TensorTrain(object):
    def __init__(self, elements):
        """
        Construct tensor train from a single numpy array, a list of cores
        as numpy arrays, or a list of list of cores as numpy arrays.
        """
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

    def _construct_scikit_tt(self, elements):
        """
        Construct TT using scikit_tt.TT objects.
        """
        if isinstance(elements, list):
            # Ensure elements are 4D
            for i in range(len(elements)):
                if isinstance(elements[i], np.ndarray):
                    # Change numpy array shape to 4d
                    if len(elements[i].shape) == 2:
                        elements[i] = elements[i][np.newaxis, :, :, np.newaxis]
                    elif len(elements[i].shape) == 3:
                        elements[i] = elements[i][:, :, np.newaxis, :]
                    elif len(elements[i].shape) != 4:
                        raise RuntimeError(
                            f"Shape of core {elements[i].shape} is not supported"
                        )
                else:
                    elements[i] = self._construct_scikit_tt(elements[i])

        else:
            return TT(elements)

        if isinstance(elements[0], np.ndarray):
            return TT(elements)

        else:
            train = elements[0]
            for array in elements[1:]:
                train += array

            return train

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

    # =======================================================================
    # Overloads

    def __add__(self, other):
        if isinstance(other, TensorTrain):
            other = other._train

        train = self._train + other
        return TensorTrain(train)

    def __sub__(self, other):
        if isinstance(other, TensorTrain):
            other = other._train

        train = self._train - other
        return TensorTrain(train)

    def __mul__(self, other):
        assert not isinstance(other, TensorTrain)

        train = self._train * other
        return TensorTrain(train)

    def __truediv__(self, other):
        assert not isinstance(other, TensorTrain)

        train = self._train * (1 / other)
        return TensorTrain(train)

    def __matmul__(self, other):
        assert isinstance(other, TensorTrain)
        return TensorTrain(self._train.dot(other.train()))

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

            return ttpy.matrix.from_list(self.cores)

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
