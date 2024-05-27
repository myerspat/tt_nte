class Nuclide:
    def __init__(self, Z, A):
        self._Z = Z
        self._A = A

    @property
    def Z(self):
        return self._Z

    @property
    def A(self):
        return self._A
