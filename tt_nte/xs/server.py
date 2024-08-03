class Server:
    def __init__(self, xs):
        self._xs = xs
        self._chi = self._xs.pop("chi")

        # Assert chi is 1D and get number of groups
        assert len(self._chi.shape) == 1
        self._num_groups = self._chi.size

        # Shape assertions
        assert self._chi.shape == (self._num_groups,)

        self._num_moments = None
        for mat in self._xs.keys():
            assert self.total(mat).shape == (self._num_groups,)
            assert self.nu_fission(mat).shape == (self._num_groups,)
            assert len(self.scatter_gtg(mat).shape) == 3
            assert self.scatter_gtg(mat).shape[1:] == (
                self._num_groups,
                self._num_groups,
            )

            if self._num_moments is None:
                self._num_moments = self.scatter_gtg(mat).shape[0]
            else:
                assert self._num_moments == self.scatter_gtg(mat).shape[0]

    def total(self, mat):
        return self._xs[mat]["total"]

    def nu_fission(self, mat):
        return self._xs[mat]["nu_fission"]

    def scatter_gtg(self, mat):
        return self._xs[mat]["scatter_gtg"]

    @property
    def chi(self):
        return self._chi

    @property
    def num_groups(self):
        return self._num_groups

    @property
    def num_moments(self):
        return self._num_moments
