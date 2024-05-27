class XSServer:
    def __init__(self, xs):
        self._xs = xs
        self._num_groups = self._xs.pop("num_groups")
        self._chi = self._xs.pop("chi")

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
